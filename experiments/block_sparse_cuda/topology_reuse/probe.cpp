// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/path.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <fstream>
#include <map>
#include <set>
#include <iostream>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Key     = std::vector<long>;
  struct Seen {
    std::map<long, long> topologies;
    std::size_t bytes = 0;
  };
  struct Limit {};
} // namespace
int main(int argc, char **argv) {
  try {
    int modes = 4, zero_every = 0;
    long count = 2, limit = 4096;
    std::string partition = "number", file;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (++i == argc) throw std::invalid_argument("Missing value");
      std::string value = argv[i];
      if (key == "--modes")
        modes = std::stoi(value);
      else if (key == "--partition")
        partition = value;
      else if (key == "--count")
        count = std::stol(value);
      else if (key == "--path-limit")
        limit = std::stol(value);
      else if (key == "--zero-every")
        zero_every = std::stoi(value);
      else if (key == "--topologies")
        file = value;
      else
        throw std::invalid_argument("Unknown option");
    }
    if (modes < 2 || modes > 6 || count < 1 || count > 4 || limit < 1 || limit > 4096 || zero_every < 0 || zero_every > 64)
      throw std::invalid_argument("Beyond bounded probe");
    std::ifstream input(file);
    long topologies = 0, order = 0;
    input >> topologies >> order;
    if (!input || topologies < 1 || topologies > 8 || order < 1 || order > 8) throw std::invalid_argument("Invalid topology cohort");
    auto ad    = block_sparse_bench::atom(modes, partition);
    auto coeff = block_sparse_bench::coefficients(ad);
    auto model = gpu::export_model(ad, coeff);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    std::vector<bool> live(G.size());
    for (long b = 0; b < G.size(); ++b) live[b] = nda::max_element(nda::abs(G[b].data())) >= 1e-16 && (!zero_every || b % zero_every);
    nda::vector<double> poles{-1.5, .6};
    std::map<std::string, std::map<Key, Seen>> stages;
    for (auto const *stage : {"root", "vertex", "edge"}) stages[stage];
    long paths     = 0;
    bool truncated = false;
    for (long topology_id = 0; topology_id < topologies; ++topology_id) {
      nda::array<int, 2> topology(order, 2);
      for (long i = 0; i < order; ++i) input >> topology(i, 0) >> topology(i, 1);
      if (!input) throw std::invalid_argument("Missing topology entries");
      auto total = triqs_xca::backbone::Backbone(topology, model.num_fields(), model.num_interactions()).num_flat_indices(model.num_poles());
      try {
        gpu::for_each_self_energy_path(model, poles, topology, live, 0, std::min(count, total), [&](auto &backbone, auto const &path) {
          if (paths == limit) throw Limit{};
          ++paths;
          long root = path.propagator_blocks[0], cols = model.dimensions()[root];
          // Fixed model/G generations, rank, layout and beta belong to every key.
          Key key{0, 0, mesh.dlr_it().rank(), 0, 2, root};
          auto record = [&](std::string const &stage, long rows) {
            auto &seen = stages[stage][key];
            ++seen.topologies[topology_id];
            seen.bytes = gpu::tensor_bytes(mesh.dlr_it().rank(), rows, cols);
          };
          record("root", cols);
          for (int vertex = 1; vertex < path.special_vertex; ++vertex) {
            int field = backbone.get_vertex_orb(vertex), pole = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(vertex));
            bool bar = backbone.has_vertex_bar(vertex), dag = backbone.has_vertex_dag(vertex);
            auto const &op    = bar ? model.barred(field, pole, dag) : model.field(field, dag);
            auto const &block = op.blocks[path.propagator_blocks[vertex - 1]];
            key.insert(key.end(), {10, field, bar, dag, bar ? pole : -1, path.propagator_blocks[vertex - 1], block.rows, block.columns});
            int sign = backbone.get_vertex_Ksign(vertex);
            if (sign) key.insert(key.end(), {11, sign, pole});
            record("vertex", block.rows);
            int source = path.propagator_blocks[vertex];
            key.insert(key.end(), {20, source, model.dimensions()[source]});
            for (int x = 0; x < backbone.m - 1; ++x) {
              int weight = backbone.get_edge(vertex, x);
              if (weight) key.insert(key.end(), {21, weight, backbone.get_pole_ind(x)});
            }
            key.push_back(22); // Same ordered solve/convolution, fixed tables/beta.
            record("edge", block.rows);
          }
          // Stop before special parity/hybridization sums. No suffix equivalence claim.
        });
      } catch (Limit const &) {
        truncated = true;
        break;
      }
    }
    std::cout
       << "stage,topologies,order,modes,partition,zero_every,visited_paths,truncated,occurrences,unique_keys,cross_topology_keys,distinct_topology_reuses,unique_payload_bytes\n";
    for (auto const &[stage, keys] : stages) {
      long occurrences = 0, cross = 0, reuses = 0;
      std::size_t bytes = 0;
      for (auto const &[key, seen] : keys) {
        (void)key;
        for (auto const &[t, n] : seen.topologies) {
          (void)t;
          occurrences += n;
        }
        if (seen.topologies.size() > 1) ++cross;
        reuses += seen.topologies.size() - 1;
        bytes = gpu::checked_add(bytes, seen.bytes);
      }
      std::cout << stage << ',' << topologies << ',' << order << ',' << modes << ',' << partition << ',' << zero_every << ',' << paths << ','
                << truncated << ',' << occurrences << ',' << keys.size() << ',' << cross << ',' << reuses << ',' << bytes << '\n';
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

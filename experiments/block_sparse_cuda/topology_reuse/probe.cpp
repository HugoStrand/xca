// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include "../dlr_hoisted/hoisted.hpp"
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/block_sparse/cuda/path.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <fstream>
#include <map>
#include <set>
#include <iostream>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Key     = std::vector<long>;
  using Array   = nda::array<nda::dcomplex, 3>;
  struct Seen {
    std::map<long, long> topologies;
    std::size_t bytes = 0;
    Array value;
  };
  struct Limit {};
} // namespace
int main(int argc, char **argv) {
  try {
    int modes = 4, zero_every = 0;
    long count = 2, limit = 4096;
    bool replay    = false;
    double g_scale = 1., coefficient_scale = 1.;
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
      else if (key == "--replay") {
        if (value != "0" && value != "1") throw std::invalid_argument("Replay must be 0 or 1");
        replay = value == "1";
      } else if (key == "--g-scale")
        g_scale = std::stod(value);
      else if (key == "--coefficient-scale")
        coefficient_scale = std::stod(value);
      else
        throw std::invalid_argument("Unknown option");
    }
    if (modes < 2 || modes > 6 || count < 1 || count > 4 || limit < 1 || limit > 4096 || zero_every < 0 || zero_every > 64)
      throw std::invalid_argument("Beyond bounded probe");
    if (!std::isfinite(g_scale) || !std::isfinite(coefficient_scale) || g_scale < .5 || g_scale > 2. || coefficient_scale < .5
        || coefficient_scale > 2.)
      throw std::invalid_argument("Beyond bounded input scaling");
    std::ifstream input(file);
    long topologies = 0, order = 0;
    input >> topologies >> order;
    if (!input || topologies < 1 || topologies > 8 || order < 1 || order > 8) throw std::invalid_argument("Invalid topology cohort");
    auto ad    = block_sparse_bench::atom(modes, partition);
    auto coeff = block_sparse_bench::coefficients(ad);
    coeff *= coefficient_scale;
    auto model = gpu::export_model(ad, coeff);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    for (long b = 0; b < G.size(); ++b) G[b].data() *= g_scale;
    std::vector<bool> live(G.size());
    for (long b = 0; b < G.size(); ++b) live[b] = nda::max_element(nda::abs(G[b].data())) >= 1e-16 && (!zero_every || b % zero_every);
    nda::vector<double> poles{-1.5, .6};
    triqs_xca::hyb::Hybridization hyb(mesh, poles, coeff);
    auto const &host = mesh.dlr_it();
    block_sparse_dlr_experiment::Hoisted hoisted(host);
    auto weight = [&](Array &value, int sign, int pole) {
      if (!sign) return;
      for (long t = 0; t < value.extent(0); ++t)
        for (long i = 0; i < value.extent(1); ++i)
          for (long j = 0; j < value.extent(2); ++j) value(t, i, j) *= sign > 0 ? hyb.k_it_p(t, pole) : hyb.k_it_m(t, pole);
    };
    struct Checks {
      long repeats       = 0;
      double reuse_error = 0., reference_error = 0., checksum = 0.;
    };
    std::map<std::string, Checks> checks;
    std::size_t retained = 0;
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
          Array value, reference;
          if (replay) {
            value     = G[root].data();
            reference = value;
          }
          auto record = [&](std::string const &stage, long rows) {
            auto &seen = stages[stage][key];
            ++seen.topologies[topology_id];
            seen.bytes = gpu::tensor_bytes(mesh.dlr_it().rank(), rows, cols);
            if (replay) {
              for (auto z : value)
                if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("Nonfinite replay value");
              for (auto z : reference)
                if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("Nonfinite independent reference");
              auto &check           = checks[stage];
              double error          = nda::max_element(nda::abs(value - reference));
              check.reference_error = std::max(check.reference_error, error);
              if (!std::isfinite(error) || error > 1e-10 * nda::max_element(nda::abs(reference)) + 1e-14)
                throw std::runtime_error("Independent prefix replay mismatch");
              if (seen.value.size()) {
                ++check.repeats;
                double reuse_error = nda::max_element(nda::abs(value - seen.value));
                check.reuse_error  = std::max(check.reuse_error, reuse_error);
                if (reuse_error != 0.) throw std::runtime_error("Shared key has unequal values");
              } else {
                retained = gpu::checked_add(retained, seen.bytes);
                if (retained > 64ULL * 1024 * 1024) throw std::length_error("Replay cache exceeds 64 MiB");
                seen.value = value;
                for (auto z : value) check.checksum += std::abs(z);
              }
            }
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
            if (replay) {
              value      = block.apply_left(value);
              auto dense = block.to_dense();
              Array next(host.rank(), block.rows, cols);
              constexpr auto all = nda::range::all;
              for (long t = 0; t < host.rank(); ++t) next(t, all, all) = cppdlr::matmul(dense, reference(t, all, all));
              reference = std::move(next);
              weight(value, sign, pole);
              weight(reference, sign, pole);
            }
            record("vertex", block.rows);
            int source = path.propagator_blocks[vertex];
            key.insert(key.end(), {20, source, model.dimensions()[source]});
            for (int x = 0; x < backbone.m - 1; ++x) {
              int edge_sign = backbone.get_edge(vertex, x);
              if (edge_sign) key.insert(key.end(), {21, edge_sign, backbone.get_pole_ind(x)});
            }
            key.push_back(22); // Same ordered solve/convolution, fixed tables/beta.
            if (replay) {
              Array weighted = G[source].data();
              for (int x = 0; x < backbone.m - 1; ++x) weight(weighted, backbone.get_edge(vertex, x), backbone.get_pole_ind(x));
              auto gc   = host.vals2coefs(weighted);
              value     = hoisted.convolve(2., gc, host.vals2coefs(value));
              reference = host.convolve(2., gc, host.vals2coefs(reference), cppdlr::TIME_ORDERED);
            }
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
       << "stage,topologies,order,modes,partition,zero_every,visited_paths,truncated,occurrences,unique_keys,cross_topology_keys,distinct_topology_reuses,unique_payload_bytes,replay,repeated_value_checks,max_reuse_error,max_reference_error,unique_value_checksum\n";
    std::cout << std::setprecision(17);
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
                << truncated << ',' << occurrences << ',' << keys.size() << ',' << cross << ',' << reuses << ',' << bytes << ',' << replay << ','
                << checks[stage].repeats << ',' << checks[stage].reuse_error << ',' << checks[stage].reference_error << ',' << checks[stage].checksum
                << '\n';
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

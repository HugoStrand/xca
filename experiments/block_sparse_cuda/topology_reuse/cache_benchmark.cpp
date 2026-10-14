// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "prefix_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <fstream>
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  try {
    if (argc != 4 && argc != 7) throw std::invalid_argument("Usage: benchmark topology_file partition library|ordered_mapped [first count modes]");
    long first   = argc == 7 ? std::stol(argv[4]) : 0;
    long indices = argc == 7 ? std::stol(argv[5]) : 2;
    int modes    = argc == 7 ? std::stoi(argv[6]) : 4;
    if (first < 0 || first > 4096 || indices < 1 || indices > 8 || modes < 2 || modes > 6) throw std::invalid_argument("Beyond bounded selection");
    std::string backend = argv[3];
    if (backend != "library" && backend != "ordered_mapped") throw std::invalid_argument("Invalid backend");
    auto sparse  = backend == "library" ? gpu::SparseBackend::Library : gpu::SparseBackend::Ordered;
    auto special = backend == "library" ? gpu::SpecialBackend::Reference : gpu::SpecialBackend::Mapped;
    std::ifstream input(argv[1]);
    int count = 0, order = 0;
    input >> count >> order;
    if (!input || count < 1 || count > 8 || order < 2 || order > 8) throw std::invalid_argument("Invalid cohort");
    std::vector<nda::array<int, 2>> topologies;
    for (int t = 0; t < count; ++t) {
      nda::array<int, 2> topology(order, 2);
      for (int i = 0; i < order; ++i) input >> topology(i, 0) >> topology(i, 1);
      if (!input) throw std::invalid_argument("Missing entries");
      topologies.push_back(std::move(topology));
    }
    auto ad           = block_sparse_bench::atom(modes, argv[2]);
    auto coefficients = block_sparse_bench::coefficients(ad);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    gpu::DiagramEvaluator reference(gpu::export_model(ad, coefficients), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    gpu::PrefixEvaluator cached(gpu::export_model(ad, coefficients), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    for (auto const &topology : topologies)
      if (first + indices > reference.get_num_self_energy_backbones(topology)) throw std::out_of_range("Selection outside backbone range");
    std::cout << std::setprecision(17)
              << "phase,sample,reference_ms,cached_ms,hits,charged_cache_bytes,max_abs_error,accepted_paths,contributing_paths\n";
    // Each cold sample includes fresh input snapshot and cache construction.
    // Replay follows the same topology order; changed inputs invalidate automatically.
    for (auto phase : {"cold", "replay", "changed_G"}) {
      for (int sample = 0; sample < 3; ++sample) {
        if (std::string(phase) == "changed_G")
          for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.01;
        if (std::string(phase) == "cold") cached.clear_prefix_cache();
        std::vector<block_sparse_bench::Gf> expected, actual;
        std::size_t accepted = 0, contributing = 0, reference_accepted = 0, reference_contributing = 0;
        auto start = block_sparse_bench::clock_type::now();
        for (auto const &topology : topologies) {
          expected.push_back(reference.compute_self_energy_range(G, topology, first, first + indices));
          reference_accepted += reference.last_path_counts().accepted;
          reference_contributing += reference.last_contributing_paths();
        }
        auto ref_ms     = block_sparse_bench::milliseconds(start);
        auto prior_hits = cached.prefix_cache_hits();
        start           = block_sparse_bench::clock_type::now();
        for (auto const &topology : topologies) {
          actual.push_back(cached.compute_self_energy_range(G, topology, first, first + indices));
          accepted += cached.last_path_counts().accepted;
          contributing += cached.last_contributing_paths();
        }
        auto cached_ms = block_sparse_bench::milliseconds(start);
        double error   = 0.;
        for (int t = 0; t < count; ++t)
          for (long b = 0; b < G.size(); ++b) {
            for (auto z : actual[t][b].data())
              if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("Nonfinite cached output");
            error = std::max(error, nda::max_element(nda::abs(expected[t][b].data() - actual[t][b].data())));
          }
        if (error != 0. || accepted != reference_accepted || contributing != reference_contributing)
          throw std::runtime_error("Cache changed complete output/path counts");
        // An input invalidation resets counters, so report the new epoch total.
        auto hits = cached.prefix_cache_hits();
        if (std::string(phase) != "changed_G") hits -= prior_hits;
        std::cout << phase << ',' << sample << ',' << ref_ms << ',' << cached_ms << ',' << hits << ',' << cached.prefix_cache_bytes() << ',' << error
                  << ',' << accepted << ',' << contributing << '\n';
      }
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

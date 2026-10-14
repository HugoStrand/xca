// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "prefix_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <fstream>
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  try {
    if (argc != 4) throw std::invalid_argument("Usage: benchmark topology_file partition library|ordered_mapped");
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
    auto ad           = block_sparse_bench::atom(4, argv[2]);
    auto coefficients = block_sparse_bench::coefficients(ad);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    gpu::DiagramEvaluator reference(gpu::export_model(ad, coefficients), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    gpu::PrefixEvaluator cached(gpu::export_model(ad, coefficients), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    std::cout << std::setprecision(17) << "phase,sample,reference_ms,cached_ms,hits,charged_cache_bytes,max_abs_error\n";
    // Each cold sample includes fresh input snapshot and cache construction.
    // Replay follows the same topology order; changed inputs invalidate automatically.
    for (auto phase : {"cold", "replay", "changed_G"}) {
      for (int sample = 0; sample < 3; ++sample) {
        if (std::string(phase) == "changed_G")
          for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.01;
        if (std::string(phase) == "cold") cached.clear_prefix_cache();
        std::vector<block_sparse_bench::Gf> expected, actual;
        auto start = block_sparse_bench::clock_type::now();
        for (auto const &topology : topologies) expected.push_back(reference.compute_self_energy_range(G, topology, 0, 2));
        auto ref_ms     = block_sparse_bench::milliseconds(start);
        auto prior_hits = cached.prefix_cache_hits();
        start           = block_sparse_bench::clock_type::now();
        for (auto const &topology : topologies) actual.push_back(cached.compute_self_energy_range(G, topology, 0, 2));
        auto cached_ms = block_sparse_bench::milliseconds(start);
        double error   = 0.;
        for (int t = 0; t < count; ++t)
          for (long b = 0; b < G.size(); ++b) { error = std::max(error, nda::max_element(nda::abs(expected[t][b].data() - actual[t][b].data()))); }
        if (error != 0.) throw std::runtime_error("Cache changed complete output");
        // An input invalidation resets counters, so report the new epoch total.
        auto hits = cached.prefix_cache_hits();
        if (std::string(phase) != "changed_G") hits -= prior_hits;
        std::cout << phase << ',' << sample << ',' << ref_ms << ',' << cached_ms << ',' << hits << ',' << cached.prefix_cache_bytes() << ',' << error
                  << '\n';
      }
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

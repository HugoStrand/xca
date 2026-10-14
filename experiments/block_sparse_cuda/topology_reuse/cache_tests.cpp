// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "prefix_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  void equal(block_sparse_bench::Gf const &actual, block_sparse_bench::Gf const &expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (long b = 0; b < actual.size(); ++b)
      for (long t = 0; t < actual[b].data().extent(0); ++t)
        for (long i = 0; i < actual[b].data().extent(1); ++i)
          for (long j = 0; j < actual[b].data().extent(2); ++j) {
            auto z = actual[b].data()(t, i, j);
            ASSERT_TRUE(std::isfinite(z.real()) && std::isfinite(z.imag()));
            ASSERT_EQ(z, expected[b].data()(t, i, j));
          }
  }
} // namespace
void check_limits(gpu::SparseBackend sparse, gpu::SpecialBackend special) {
  for (auto partition : {"number", "mixed", "occupation"}) {
    auto ad           = block_sparse_bench::atom(4, partition);
    auto coefficients = block_sparse_bench::coefficients(ad);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    gpu::DiagramEvaluator reference(gpu::export_model(ad, coefficients), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    gpu::PrefixEvaluator cached(gpu::export_model(ad, coefficients), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, sparse, special);
    nda::array<int, 2> topology{{0, 3}, {1, 4}, {2, 5}};
    for (int control = 0; control < 4; ++control) {
      // Byte rejection, one-entry saturation, host snapshot bypass, normal cache.
      std::size_t bytes    = control == 0 ? 1 : 8ULL * 1024 * 1024;
      std::size_t entries  = control == 1 ? 1 : 4096;
      std::size_t snapshot = control == 2 ? 1 : 32ULL * 1024 * 1024;
      cached.configure_prefix_cache(bytes, entries, snapshot);
      for (int replay = 0; replay < 2; ++replay)
        for (auto range : {std::pair{0L, 1L}, std::pair{1L, 3L}, std::pair{3L, 4L}}) {
          SCOPED_TRACE(partition);
          SCOPED_TRACE(control);
          equal(cached.compute_self_energy_range(G, topology, range.first, range.second),
                reference.compute_self_energy_range(G, topology, range.first, range.second));
          EXPECT_LE(cached.prefix_cache_bytes(), bytes);
          EXPECT_LE(cached.prefix_cache_entries(), entries);
        }
      if (control == 0 || control == 2) {
        EXPECT_EQ(cached.prefix_cache_entries(), 0);
        EXPECT_EQ(cached.prefix_cache_hits(), 0);
      }
      if (control == 2) EXPECT_EQ(cached.prefix_snapshot_bytes(), 0);
      if (control == 1 && std::string(partition) == "number") {
        EXPECT_EQ(cached.prefix_cache_entries(), 1);
        EXPECT_GT(cached.prefix_cache_hits(), 0);
        EXPECT_GT(cached.prefix_cache_skipped(), 0);
      }
      // Invalid configuration must preserve populated state.
      auto retained = cached.prefix_cache_bytes();
      EXPECT_THROW(cached.configure_prefix_cache(8ULL * 1024 * 1024 + 1, 1, 1), std::invalid_argument);
      EXPECT_EQ(cached.prefix_cache_bytes(), retained);
      equal(cached.compute_self_energy_range(G, topology, 2, 2), reference.compute_self_energy_range(G, topology, 2, 2));
      EXPECT_EQ(cached.prefix_cache_bytes(), retained);
      auto changed = G;
      for (long b = 0; b < changed.size(); ++b) changed[b].data() *= 1.01;
      equal(cached.compute_self_energy_range(changed, topology, 0, 4), reference.compute_self_energy_range(changed, topology, 0, 4));
      for (long b = 0; b < changed.size(); ++b)
        if (b % 2 == 0) changed[b].data() = 0.;
      equal(cached.compute_self_energy_range(changed, topology, 0, 4), reference.compute_self_energy_range(changed, topology, 0, 4));
      // Restore a previously used input after invalidation.
      equal(cached.compute_self_energy_range(G, topology, 0, 4), reference.compute_self_energy_range(G, topology, 0, 4));
      cached.clear_prefix_cache();
      EXPECT_EQ(cached.prefix_cache_entries(), 0);
      EXPECT_EQ(cached.prefix_snapshot_bytes(), 0);
    }
  }
}

TEST(BlockSparsePrefixCache, SaturationTailsAndInputChanges) { check_limits(gpu::SparseBackend::Library, gpu::SpecialBackend::Reference); }
TEST(BlockSparsePrefixCache, OrderedMappedSaturation) {
  if (!gpu::ordered_sparse_available() || !gpu::mapped_special_available()) GTEST_SKIP() << "Native kernels unavailable";
  check_limits(gpu::SparseBackend::Ordered, gpu::SpecialBackend::Mapped);
}

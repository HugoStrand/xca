// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "row_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
namespace {
  namespace gpu             = triqs_xca::block_sparse::cuda;
  constexpr std::size_t GiB = 1024ULL * 1024 * 1024, sparse = 32ULL * 1024 * 1024;
  void compare(auto const &actual, auto const &expected) {
    auto norm  = nda::max_element(nda::abs(expected));
    auto error = nda::max_element(nda::abs(actual - expected));
    for (auto z : actual) ASSERT_TRUE(std::isfinite(z.real()) && std::isfinite(z.imag()));
    EXPECT_LE(error, 1e-13 * std::max(norm, 1e-20));
  }
  void counts(auto const &a, auto const &b) {
    EXPECT_EQ(a.last_path_counts().proposed, b.last_path_counts().proposed);
    EXPECT_EQ(a.last_path_counts().accepted, b.last_path_counts().accepted);
    EXPECT_EQ(a.last_path_counts().pruned, b.last_path_counts().pruned);
    EXPECT_EQ(a.last_contributing_paths(), b.last_contributing_paths());
  }
} // namespace
TEST(RowEvaluator, CompleteCallsChangedInputAndPlacement) {
  for (auto partition : {"number", "mixed"}) {
    auto ad    = block_sparse_bench::atom(4, partition);
    auto coeff = block_sparse_bench::coefficients(ad);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    auto model = gpu::export_model(ad, coeff);
    nda::array<int, 2> top{{0, 2}, {1, 3}};
    gpu::DiagramEvaluator ref(model, poles, mesh);
    for (auto placement : {gpu::OutputPlacement::Resident, gpu::OutputPlacement::HostSectors}) {
      gpu::RowDiagramEvaluator row(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference, placement, GiB,
                                   gpu::NcaBackend::Reference, 1);
      for (int update = 0; update < 3; ++update) {
        if (update == 1)
          for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.017;
        if (update == 2)
          for (long b = 0; b < G.size(); ++b)
            if (b % 2 == 0) G[b].data() = 0.;
        auto expected = ref.compute_self_energy_range(G, top, 0, 2);
        auto actual   = row.compute_self_energy_range(G, top, 0, 2);
        for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
        counts(row, ref);
        auto again = row.compute_self_energy_range(G, top, 0, 2);
        for (long b = 0; b < G.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(again[b].data() - actual[b].data())), 0.);
        auto reff  = ref.compute_single_ptcle_gf_range(G, top, 0, 2);
        auto value = row.compute_single_ptcle_gf_range(G, top, 0, 2);
        compare(value, reff);
        counts(row, ref);
      }
    }
  }
}
TEST(RowEvaluator, CompleteBudgetRejectionAndRecovery) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, coeff);
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  gpu::RowDiagramEvaluator estimate(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                    gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference, 1);
  gpu::DiagramEvaluator ref(model, poles, mesh);
  auto row_bound = estimate.self_energy_storage_bound(oca);
  EXPECT_LT(row_bound, ref.self_energy_storage_bound(oca));
  EXPECT_EQ(estimate.self_energy_storage_bound(nca), ref.self_energy_storage_bound(nca));
  gpu::RowDiagramEvaluator tight(model, poles, mesh, row_bound, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                 gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference, 1);
  gpu::RowDiagramEvaluator rejected(model, poles, mesh, row_bound - 1, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                    gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference, 1);
  EXPECT_THROW(rejected.compute_self_energy_range(G, oca, 0, 1), std::length_error);
  tight.sample_row_allocations(true);
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto expected = ref.compute_self_energy_range(G, oca, 0, 1);
    auto actual   = tight.compute_self_energy_range(G, oca, 0, 1);
    for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
    // Driver allocation caching/granularity can yield a zero global free-byte delta.
    auto nref      = ref.compute_self_energy_range(G, nca, 0, 1);
    auto recovered = rejected.compute_self_energy_range(G, nca, 0, 1);
    for (long b = 0; b < G.size(); ++b) compare(recovered[b].data(), nref[b].data());
  }
  EXPECT_THROW((gpu::RowDiagramEvaluator(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                         gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference, 0)),
               std::invalid_argument);
}

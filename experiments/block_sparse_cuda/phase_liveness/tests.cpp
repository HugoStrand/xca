// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "lifetime_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
namespace {
  namespace gpu             = triqs_xca::block_sparse::cuda;
  constexpr std::size_t GiB = 1024ULL * 1024 * 1024, sparse = 32ULL * 1024 * 1024;
  void compare(auto const &actual, auto const &expected) {
    auto error = nda::max_element(nda::abs(actual - expected));
    for (auto z : actual) ASSERT_TRUE(std::isfinite(z.real()) && std::isfinite(z.imag()));
    EXPECT_EQ(error, 0.);
  }
  void counts(auto const &a, auto const &b) {
    EXPECT_EQ(a.last_path_counts().proposed, b.last_path_counts().proposed);
    EXPECT_EQ(a.last_path_counts().accepted, b.last_path_counts().accepted);
    EXPECT_EQ(a.last_path_counts().pruned, b.last_path_counts().pruned);
    EXPECT_EQ(a.last_contributing_paths(), b.last_contributing_paths());
  }
} // namespace
TEST(LifetimeEvaluator, CompleteCallsChangedInputAndPlacement) {
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
      gpu::LifetimeDiagramEvaluator row(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference, placement, GiB,
                                        gpu::NcaBackend::Reference);
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
TEST(LifetimeEvaluator, CompleteBudgetRejectionAndRecovery) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, coeff);
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  gpu::LifetimeDiagramEvaluator estimate(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                         gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference);
  gpu::DiagramEvaluator ref(model, poles, mesh);
  auto row_bound = estimate.self_energy_storage_bound(oca);
  EXPECT_LT(row_bound, ref.self_energy_storage_bound(oca));
  auto tile = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
  EXPECT_EQ(ref.self_energy_storage_bound(oca) - row_bound, 5 * tile);
  EXPECT_EQ(estimate.self_energy_storage_bound(nca), ref.self_energy_storage_bound(nca));
  gpu::LifetimeDiagramEvaluator tight(model, poles, mesh, row_bound, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                      gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference);
  gpu::LifetimeDiagramEvaluator rejected(model, poles, mesh, row_bound - 1, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                         gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference);
  EXPECT_THROW(rejected.compute_self_energy_range(G, oca, 0, 1), std::length_error);
  gpu::DiagramEvaluator original_rejected(model, poles, mesh, row_bound);
  EXPECT_THROW(original_rejected.compute_self_energy_range(G, oca, 0, 1), std::length_error);
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto expected = ref.compute_self_energy_range(G, oca, 0, 1);
    auto actual   = tight.compute_self_energy_range(G, oca, 0, 1);
    for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
    auto nref      = ref.compute_self_energy_range(G, nca, 0, 1);
    auto recovered = rejected.compute_self_energy_range(G, nca, 0, 1);
    for (long b = 0; b < G.size(); ++b) compare(recovered[b].data(), nref[b].data());
  }
}
TEST(LifetimeEvaluator, CorrelatorBudgetAndEndpointRecovery) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, coeff);
  nda::array<int, 2> top{{0, 2}, {1, 3}};
  gpu::LifetimeDiagramEvaluator estimate(model, poles, mesh);
  gpu::DiagramEvaluator reference(model, poles, mesh);
  auto n     = model.num_hybridization_fields();
  auto bound = estimate.correlator_storage_bound(n, n);
  auto tile  = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
  EXPECT_EQ(reference.correlator_storage_bound(n, n) - bound, 5 * tile);
  gpu::LifetimeDiagramEvaluator tight(model, poles, mesh, bound);
  gpu::LifetimeDiagramEvaluator rejected(model, poles, mesh, bound - 1);
  gpu::DiagramEvaluator old_rejected(model, poles, mesh, bound);
  EXPECT_THROW(rejected.compute_single_ptcle_gf_range(G, top, 0, 1), std::length_error);
  EXPECT_THROW(old_rejected.compute_single_ptcle_gf_range(G, top, 0, 1), std::length_error);
  for (int update = 0; update < 2; ++update) {
    if (update)
      for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.01;
    compare(tight.compute_single_ptcle_gf_range(G, top, 0, 1), reference.compute_single_ptcle_gf_range(G, top, 0, 1));
    counts(tight, reference);
    std::vector<gpu::SparseOperator> empty;
    auto zero = rejected.compute_correlator_range(G, empty, empty, false, top, 0, 1);
    EXPECT_EQ(zero.size(), 0);
    auto before          = G[0].data()(0, 0, 0);
    G[0].data()(0, 0, 0) = {std::numeric_limits<double>::quiet_NaN(), 0.};
    EXPECT_THROW(tight.compute_single_ptcle_gf_range(G, top, 0, 1), std::invalid_argument);
    G[0].data()(0, 0, 0) = before;
    compare(tight.compute_single_ptcle_gf_range(G, top, 0, 1), reference.compute_single_ptcle_gf_range(G, top, 0, 1));
  }
}

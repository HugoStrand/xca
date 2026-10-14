// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "stream_evaluator.hpp"
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
TEST(StreamFragmentEvaluator, CompleteCallsChangedInputAndPlacement) {
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
      gpu::StreamFragmentEvaluator row(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference, placement, GiB,
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
TEST(StreamFragmentEvaluator, CompleteBudgetRejectionAndRecovery) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, coeff);
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  gpu::StreamFragmentEvaluator estimate(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                        gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference);
  gpu::DiagramEvaluator ref(model, poles, mesh);
  auto row_bound = estimate.self_energy_storage_bound(oca);
  EXPECT_LT(row_bound, ref.self_energy_storage_bound(oca));
  auto tile = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
  EXPECT_EQ(ref.self_energy_storage_bound(oca) - row_bound, 5 * tile + gpu::function_bytes(mesh.dlr_it().rank(), model.dimensions()));
  EXPECT_EQ(ref.self_energy_storage_bound(nca) - estimate.self_energy_storage_bound(nca),
            gpu::function_bytes(mesh.dlr_it().rank(), model.dimensions()));
  gpu::StreamFragmentEvaluator tight(model, poles, mesh, row_bound, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                     gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::Reference);
  gpu::StreamFragmentEvaluator rejected(model, poles, mesh, row_bound - 1, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
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
TEST(StreamFragmentEvaluator, CorrelatorBudgetAndEndpointRecovery) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, coeff);
  nda::array<int, 2> top{{0, 2}, {1, 3}};
  gpu::StreamFragmentEvaluator estimate(model, poles, mesh);
  gpu::DiagramEvaluator reference(model, poles, mesh);
  auto n     = model.num_hybridization_fields();
  auto bound = estimate.correlator_storage_bound(n, n);
  auto tile  = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
  EXPECT_EQ(reference.correlator_storage_bound(n, n) - bound, 4 * tile + 3 * gpu::function_bytes(mesh.dlr_it().rank(), model.dimensions()));
  gpu::StreamFragmentEvaluator tight(model, poles, mesh, bound);
  gpu::StreamFragmentEvaluator rejected(model, poles, mesh, bound - 1);
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
TEST(StreamFragmentEvaluator, CacheReuseAndPerCallInvalidation) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, coeff);
  nda::array<int, 2> top{{0, 2}, {1, 3}};
  auto endpoint = gpu::export_operator(ad, triqs::operators::n("f", 0) + .2);
  std::vector<gpu::SparseOperator> mu(2, endpoint), kap(3, endpoint);
  gpu::StreamFragmentEvaluator candidate(model, poles, mesh);
  gpu::DiagramEvaluator reference(model, poles, mesh);
  for (long index : {0, 1, 0}) {
    auto expected = reference.compute_correlator_range(G, mu, kap, false, top, index, index + 1);
    auto actual   = candidate.compute_correlator_range(G, mu, kap, false, top, index, index + 1);
    compare(actual, expected);
    counts(candidate, reference);
    auto builds = candidate.last_fragment_builds();
    ASSERT_GT(builds, 0);
    EXPECT_LT(builds, 2 * candidate.last_path_counts().accepted);
    auto repeat = candidate.compute_correlator_range(G, mu, kap, false, top, index, index + 1);
    compare(repeat, actual);
    EXPECT_EQ(candidate.last_fragment_builds(), builds);
    for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.01;
  }
  auto empty = candidate.compute_correlator_range(G, mu, kap, false, top, 0, 0);
  EXPECT_EQ(nda::max_element(nda::abs(empty)), 0.);
  EXPECT_EQ(candidate.last_fragment_builds(), 0);
  nda::array<int, 2> nca{{0, 1}};
  candidate.compute_self_energy_range(G, nca, 0, 1);
  EXPECT_EQ(candidate.last_fragment_builds(), 0);
}
TEST(StreamFragmentEvaluator, RejectsResidentArenaPolicy) {
  auto ad = block_sparse_bench::atom(4, "number");
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  EXPECT_THROW((gpu::StreamFragmentEvaluator(model, poles, mesh, GiB, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                             gpu::OutputPlacement::Resident, GiB, gpu::NcaBackend::RealBatch)),
               std::invalid_argument);
}
TEST(StreamFragmentEvaluator, NativePolicyCombinations) {
  if (!gpu::ordered_sparse_available() || !gpu::mapped_special_available()) GTEST_SKIP() << "Native policies unavailable";
  auto ad = block_sparse_bench::atom(4, "number");
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  using namespace triqs::operators;
  std::vector<many_body_operator_complex> interactions{n("f", 0) + n("f", 1) + n("f", 2) + n("f", 3), c_dag("f", 0) * c("f", 1)};
  auto dyn     = nda::zeros<nda::dcomplex>(2, 2, 2);
  dyn(0, 0, 0) = .03;
  dyn(1, 1, 1) = .02;
  auto model   = gpu::export_model(ad, block_sparse_bench::coefficients(ad), interactions, dyn);
  for (auto sparse_backend : {gpu::SparseBackend::Library, gpu::SparseBackend::Ordered})
    for (auto special : {gpu::SpecialBackend::Reference, gpu::SpecialBackend::Mapped}) {
      gpu::StreamFragmentEvaluator candidate(model, poles, mesh, GiB, sparse, sparse_backend, special);
      gpu::DiagramEvaluator reference(model, poles, mesh, GiB, sparse, sparse_backend, special);
      for (auto top : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}}) {
        auto se_total = reference.get_num_self_energy_backbones(top);
        auto gf_total = reference.get_num_single_ptcle_gf_backbones(top);
        for (long index : {0L, se_total / 2, se_total - 1}) {
          auto expected = reference.compute_self_energy_range(G, top, index, index + 1);
          auto actual   = candidate.compute_self_energy_range(G, top, index, index + 1);
          for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
          counts(candidate, reference);
        }
        for (long index : {0L, gf_total / 2, gf_total - 1}) {
          compare(candidate.compute_single_ptcle_gf_range(G, top, index, index + 1),
                  reference.compute_single_ptcle_gf_range(G, top, index, index + 1));
          counts(candidate, reference);
        }
      }
    }
}

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <numeric>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "reference_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
namespace gpu = triqs_xca::block_sparse::cuda;
namespace {
  constexpr std::size_t budget = 1ULL << 30, sparse = 32ULL << 20;
  void compare(auto const &a, auto const &b) { EXPECT_EQ(nda::max_element(nda::abs(a - b)), 0.); }
  void counts(auto const &a, auto const &b) {
    EXPECT_EQ(a.last_path_counts().proposed, b.last_path_counts().proposed);
    EXPECT_EQ(a.last_path_counts().accepted, b.last_path_counts().accepted);
    EXPECT_EQ(a.last_path_counts().pruned, b.last_path_counts().pruned);
    EXPECT_EQ(a.last_contributing_paths(), b.last_contributing_paths());
  }
} // namespace
TEST(StoragePolicy, CompleteCallsBoundsAndRecovery) {
  for (auto partition : {"occupation", "number", "mixed"}) {
    auto ad    = block_sparse_bench::atom(4, partition);
    auto coeff = block_sparse_bench::coefficients(ad);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    auto model = gpu::export_model(ad, coeff);
    for (auto policy : {gpu::StoragePolicy::Resident, gpu::StoragePolicy::HostInputs, gpu::StoragePolicy::StreamFragments})
      for (auto placement : {gpu::OutputPlacement::Resident, gpu::OutputPlacement::HostSectors}) {
        gpu::ReferenceDiagramEvaluator reference(model, poles, mesh, budget, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                                 placement, budget);
        gpu::DiagramEvaluator candidate(model, poles, mesh, budget, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference, placement,
                                        budget, gpu::NcaBackend::Reference, policy);
        auto plan = gpu::reference_memory_plan(mesh.dlr_it().rank(), model.dimensions(), poles.size(), sparse, 0,
                                               placement == gpu::OutputPlacement::HostSectors, 4, 4, policy);
        nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
        EXPECT_EQ(candidate.self_energy_storage_bound(nca), plan.device_nca);
        EXPECT_EQ(candidate.self_energy_storage_bound(oca), plan.device_convolution);
        EXPECT_EQ(candidate.correlator_storage_bound(4, 4), plan.device_correlator);
        for (auto top : {nca, oca})
          for (int update = 0; update < 3; ++update) {
            if (update == 0) G = block_sparse_bench::propagator(ad, mesh);
            if (update == 1)
              for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.01;
            if (update == 2)
              for (long b = 0; b < G.size(); ++b)
                if (b % 2 == 0) G[b].data() = 0.;
            auto expected = reference.compute_self_energy_range(G, top, 0, std::min(2L, reference.get_num_self_energy_backbones(top))),
                 actual   = candidate.compute_self_energy_range(G, top, 0, std::min(2L, reference.get_num_self_energy_backbones(top)));
            for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
            counts(candidate, reference);
            compare(candidate.compute_single_ptcle_gf_range(G, top, 0, std::min(2L, reference.get_num_single_ptcle_gf_backbones(top))),
                    reference.compute_single_ptcle_gf_range(G, top, 0, std::min(2L, reference.get_num_single_ptcle_gf_backbones(top))));
            counts(candidate, reference);
          }
      }
  }
}
TEST(StoragePolicy, TightBudgetAndInvalidPolicies) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, coeff);
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  auto make = [&](std::size_t bytes, gpu::StoragePolicy policy, gpu::NcaBackend nca = gpu::NcaBackend::Reference) {
    return std::make_unique<gpu::DiagramEvaluator>(model, poles, mesh, bytes, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                                   gpu::OutputPlacement::Resident, budget, nca, policy);
  };
  EXPECT_THROW(make(budget, static_cast<gpu::StoragePolicy>(99)), std::invalid_argument);
  EXPECT_THROW(make(budget, gpu::StoragePolicy::HostInputs, gpu::NcaBackend::RealBatch), std::invalid_argument);
  EXPECT_THROW(make(budget, gpu::StoragePolicy::StreamFragments, gpu::NcaBackend::RealBatch), std::invalid_argument);
  for (auto policy : {gpu::StoragePolicy::HostInputs, gpu::StoragePolicy::StreamFragments}) {
    auto estimate = make(budget, policy);
    auto bytes    = estimate->self_energy_storage_bound(oca);
    auto tight = make(bytes, policy), reject = make(bytes - 1, policy), old = make(bytes, gpu::StoragePolicy::Resident);
    EXPECT_THROW(reject->compute_self_energy_range(G, oca, 0, 1), std::length_error);
    EXPECT_THROW(old->compute_self_energy_range(G, oca, 0, 1), std::length_error);
    auto expected = estimate->compute_self_energy_range(G, oca, 0, 1), actual = tight->compute_self_energy_range(G, oca, 0, 1);
    for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
    reject->compute_self_energy_range(G, nca, 0, 1);
    auto cb = estimate->correlator_storage_bound(4, 4);
    auto c  = make(cb, policy);
    compare(c->compute_single_ptcle_gf_range(G, oca, 0, 1), estimate->compute_single_ptcle_gf_range(G, oca, 0, 1));
  }
}
TEST(StoragePolicy, StridedHostInput) {
  auto ad = block_sparse_bench::atom(4, "number");
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  std::vector<nda::array<nda::dcomplex, 3>> owners;
  std::vector<triqs::gfs::gf_view<triqs::mesh::dlr_imtime>> views;
  for (long b = 0; b < G.size(); ++b) {
    long d = model.dimensions()[b];
    owners.emplace_back(mesh.dlr_it().rank(), d, 2 * d);
    owners.back() = nda::dcomplex{std::numeric_limits<double>::quiet_NaN(), 0.};
    auto data     = owners.back()(nda::range::all, nda::range::all, nda::range(0, 2 * d, 2));
    data          = G[b].data();
    views.emplace_back(mesh, data);
  }
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> strided(G.block_names(), views);
  gpu::ReferenceDiagramEvaluator reference(model, poles, mesh);
  nda::array<int, 2> top{{0, 2}, {1, 3}};
  for (auto policy : {gpu::StoragePolicy::HostInputs, gpu::StoragePolicy::StreamFragments}) {
    gpu::DiagramEvaluator candidate(model, poles, mesh, budget, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                    gpu::OutputPlacement::Resident, budget, gpu::NcaBackend::Reference, policy);
    auto expected = reference.compute_self_energy_range(G, top, 0, 1), actual = candidate.compute_self_energy_range(strided, top, 0, 1);
    for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
    compare(candidate.compute_single_ptcle_gf_range(strided, top, 0, 1), reference.compute_single_ptcle_gf_range(G, top, 0, 1));
  }
}
TEST(StoragePolicy, NativePolicyCombinations) {
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
  for (auto policy : {gpu::StoragePolicy::HostInputs, gpu::StoragePolicy::StreamFragments})
    for (auto sparse_backend : {gpu::SparseBackend::Library, gpu::SparseBackend::Ordered})
      for (auto special : {gpu::SpecialBackend::Reference, gpu::SpecialBackend::Mapped}) {
        gpu::DiagramEvaluator candidate(model, poles, mesh, budget, sparse, sparse_backend, special, gpu::OutputPlacement::Resident, budget,
                                        gpu::NcaBackend::Reference, policy);
        gpu::ReferenceDiagramEvaluator reference(model, poles, mesh, budget, sparse, sparse_backend, special);
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

TEST(StoragePolicy, AdjacentCutoffMatchesFrozenReference) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> top{{0, 2}, {1, 3}};
  for (double lambda : {20., 40.}) {
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, lambda, 1e-10, false);
    auto seed = block_sparse_bench::propagator(ad, mesh), G = seed;
    auto model = gpu::export_model(ad, coeff);
    gpu::ReferenceDiagramEvaluator reference(model, poles, mesh);
    auto eval = [&](auto &e, double scale) {
      for (long b = 0; b < G.size(); ++b) G[b].data() = scale * seed[b].data();
      return e.compute_self_energy_range(G, top, 0, 1);
    };
    double lo = 0., hi = 1.;
    eval(reference, hi);
    ASSERT_GT(reference.last_contributing_paths(), 0);
    for (int step = 0; step < 70; ++step) {
      double mid = std::midpoint(lo, hi);
      if (mid == lo || mid == hi) break;
      eval(reference, mid);
      if (reference.last_contributing_paths())
        hi = mid;
      else
        lo = mid;
    }
    ASSERT_EQ(std::nextafter(lo, 1.), hi);
    for (auto policy : {gpu::StoragePolicy::Resident, gpu::StoragePolicy::HostInputs, gpu::StoragePolicy::StreamFragments}) {
      gpu::DiagramEvaluator candidate(model, poles, mesh, budget, sparse, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                      gpu::OutputPlacement::Resident, budget, gpu::NcaBackend::Reference, policy);
      for (int neighbor = -4; neighbor <= 4; ++neighbor) {
        double scale = hi;
        for (int k = 0; k < std::abs(neighbor); ++k) scale = std::nextafter(scale, neighbor < 0 ? 0. : 1.);
        auto expected = eval(reference, scale), actual = eval(candidate, scale);
        for (long b = 0; b < G.size(); ++b) compare(actual[b].data(), expected[b].data());
        counts(candidate, reference);
      }
    }
  }
}
#else
TEST(StoragePolicy, DISABLED_RequiresCuda) {}
#endif

TEST(StoragePolicyInventory, AvailableWithoutDevice) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  std::vector<long> dims{1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1};
  auto resident = gpu::reference_memory_plan(64, dims, 2, 32ULL << 20, 0, true, 14, 14);
  auto host     = gpu::reference_memory_plan(64, dims, 2, 32ULL << 20, 0, true, 14, 14, gpu::StoragePolicy::HostInputs);
  auto streamed = gpu::reference_memory_plan(64, dims, 2, 32ULL << 20, 0, true, 14, 14, gpu::StoragePolicy::StreamFragments);
  EXPECT_EQ(resident.function, 16ULL * 64 * 40116600ULL);
  EXPECT_LT(host.device_convolution, resident.device_convolution);
  EXPECT_LT(streamed.device_correlator, host.device_correlator);
  EXPECT_EQ(streamed.device_nca, host.device_nca);
  EXPECT_EQ(streamed.host_output, resident.host_output);
  EXPECT_THROW(gpu::reference_memory_plan(64, dims, 2, 32ULL << 20, 0, false, 0, 0, static_cast<gpu::StoragePolicy>(99)), std::invalid_argument);
}

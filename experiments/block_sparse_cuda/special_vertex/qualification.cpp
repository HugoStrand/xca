// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kappa_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <gtest/gtest.h>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  void exact(block_sparse_bench::Gf const &a, block_sparse_bench::Gf const &b) {
    ASSERT_EQ(a.size(), b.size());
    for (long block = 0; block < a.size(); ++block) EXPECT_EQ(nda::max_element(nda::abs(a[block].data() - b[block].data())), 0.);
  }
}
TEST(KappaReuse, MultiFieldRectangularPathsChangedInputAndMasks) {
  auto ad = block_sparse_bench::atom(4, "number");
  auto coeff = block_sparse_bench::coefficients(ad);
  auto model = gpu::export_model(ad, coeff);
  ASSERT_TRUE(std::any_of(model.sets().begin(), model.sets().end(), [](auto const &s) { return s.size() == 4; }));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  for (auto backend : {gpu::SparseBackend::Library, gpu::SparseBackend::Ordered}) {
    if (backend == gpu::SparseBackend::Ordered && !gpu::ordered_sparse_available()) continue;
    gpu::DiagramEvaluator control(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    gpu::KappaCacheEvaluator candidate(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    for (int update = 0; update < 3; ++update) {
      if (update == 1) for (long b = 0; b < G.size(); ++b) G[b].data() *= nda::dcomplex{.7, .2};
      if (update == 2) G[2].data() = 0.;
      exact(candidate.compute_self_energy_range(G, topology, 0, 4), control.compute_self_energy_range(G, topology, 0, 4));
      EXPECT_EQ(candidate.last_contributing_paths(), control.last_contributing_paths());
      EXPECT_EQ(candidate.last_path_counts().accepted, control.last_path_counts().accepted);
    }
  }
}
TEST(KappaReuse, CacheStorageAdmittedBeforeSubmission) {
  auto ad = block_sparse_bench::atom(4, "number");
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  gpu::DiagramEvaluator control(model, poles, mesh);
  gpu::KappaCacheEvaluator candidate(model, poles, mesh);
  EXPECT_EQ(candidate.self_energy_storage_bound() - control.self_energy_storage_bound(),
            std::min(4, XCA_EXPERIMENT_KAPPA_SLOTS) * mesh.dlr_it().rank() * 6 * 6 * sizeof(nda::dcomplex));
  gpu::KappaCacheEvaluator limited(model, poles, mesh, candidate.self_energy_storage_bound(topology) - 1);
  EXPECT_THROW(limited.compute_self_energy_range(G, topology, 0, 1), std::length_error);
  EXPECT_EQ(limited.last_path_counts().accepted, 0);
  EXPECT_EQ(limited.last_device_peak_delta(), 0);
  auto empty = limited.compute_self_energy_range(G, topology, 0, 0);
  for (long b = 0; b < empty.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(empty[b].data())), 0.);
  exact(candidate.compute_self_energy(G, topology), control.compute_self_energy(G, topology));
}
TEST(KappaReuse, BosonicOffDiagonalMixingPreservesReflectedOrder) {
  auto ad = block_sparse_bench::atom(2, "occupation");
  auto coeff = block_sparse_bench::coefficients(ad);
  using triqs::operators::n;
  std::vector<triqs::operators::many_body_operator_complex> operators{n("f", 0) + 1e-8, n("f", 1) + 1e-8};
  auto dyn = nda::zeros<nda::dcomplex>(2, 2, 2);
  dyn(0, 0, 0) = .03;
  dyn(1, 1, 1) = .02;
  dyn(0, 0, 1) = {.007, .004};
  dyn(0, 1, 0) = std::conj(dyn(0, 0, 1));
  dyn(1, 0, 1) = {.003, -.006};
  dyn(1, 1, 0) = std::conj(dyn(1, 0, 1));
  auto model = gpu::export_model(ad, coeff, operators, dyn);
  ASSERT_TRUE(std::any_of(model.sets().begin(), model.sets().end(), [](auto const &s) { return s.size() == 2 && s[0] >= 2; }));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  for (auto backend : {gpu::SparseBackend::Library, gpu::SparseBackend::Ordered}) {
    if (backend == gpu::SparseBackend::Ordered && !gpu::ordered_sparse_available()) continue;
    gpu::DiagramEvaluator control(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    gpu::KappaCacheEvaluator candidate(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, backend);
    for (int update = 0; update < 2; ++update) {
      if (update) for (long b = 0; b < G.size(); ++b) G[b].data() *= nda::dcomplex{.8, -.1};
      exact(candidate.compute_self_energy(G, topology), control.compute_self_energy(G, topology));
      EXPECT_EQ(candidate.last_path_counts().accepted, control.last_path_counts().accepted);
      EXPECT_EQ(candidate.last_contributing_paths(), control.last_contributing_paths());
    }
  }
}

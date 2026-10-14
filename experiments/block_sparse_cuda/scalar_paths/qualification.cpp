// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "scalar_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <gtest/gtest.h>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using block_sparse_bench::Gf;
  double norm(Gf const &G) {
    double out = 0;
    for (long b = 0; b < G.size(); ++b)
      for (auto z : G[b].data()) out = std::max(out, std::abs(z));
    return out;
  }
  void compare(Gf const &a, Gf const &b) {
    ASSERT_EQ(a.size(), b.size());
    double error = 0;
    for (long block = 0; block < a.size(); ++block) error = std::max(error, nda::max_element(nda::abs(a[block].data() - b[block].data())));
    EXPECT_LE(error, 1e-13 * std::max(norm(a), norm(b)) + 1e-30);
  }
} // namespace
TEST(ScalarPathQualification, CohortTailChangedValuesAndMasks) {
  auto ad    = block_sparse_bench::atom(6, "occupation");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G     = block_sparse_bench::propagator(ad, mesh);
  auto model = gpu::export_model(ad, coeff);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  gpu::DiagramEvaluator reference(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
  gpu::ScalarPathEvaluator candidate(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
  auto full = candidate.compute_self_energy_range(G, topology, 0, 8);
  EXPECT_GT(candidate.last_path_counts().accepted, 512);
  EXPECT_NE(candidate.last_path_counts().accepted % 512, 0);
  auto expected = reference.compute_self_energy_range(G, topology, 0, 8);
  compare(full, expected);
  EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
  auto first = candidate.compute_self_energy_range(G, topology, 0, 3), second = candidate.compute_self_energy_range(G, topology, 3, 8);
  for (long b = 0; b < first.size(); ++b) first[b].data() += second[b].data();
  compare(first, full);
  for (long b = 0; b < G.size(); ++b) G[b].data() *= nda::dcomplex{.7, .2};
  G[7].data() = 0.;
  compare(candidate.compute_self_energy_range(G, topology, 0, 8), reference.compute_self_energy_range(G, topology, 0, 8));
  EXPECT_EQ(candidate.last_path_counts().accepted, reference.last_path_counts().accepted);
  EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
  for (long b = 0; b < G.size(); ++b) G[b].data() = 0.;
  EXPECT_EQ(norm(candidate.compute_self_energy_range(G, topology, 0, 8)), 0.);
  EXPECT_EQ(candidate.last_path_counts().accepted, 0);
  EXPECT_EQ(candidate.last_contributing_paths(), 0);
}
TEST(ScalarPathQualification, CompletedPathCutoffAdjacentValues) {
  auto ad = block_sparse_bench::atom(2, "occupation");
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 10., 1e-10, false);
  auto G         = block_sparse_bench::propagator(ad, mesh);
  auto coeff     = nda::zeros<nda::dcomplex>(1, 2, 2);
  coeff(0, 0, 0) = 1.;
  coeff(0, 1, 1) = 1.;
  auto model     = gpu::export_model(ad, coeff);
  nda::vector<double> poles{0.};
  nda::array<int, 2> topology{{0, 1}};
  gpu::DiagramEvaluator reference(model, poles, mesh);
  gpu::ScalarPathEvaluator candidate(model, poles, mesh);
  triqs_xca::hyb::Hybridization hyb(mesh, poles, coeff);
  double factor = std::max(nda::max_element(nda::abs(hyb.values)), nda::max_element(nda::abs(hyb.values_reflect)));
  // Use actual reconstructed hybridization values, not an assumed exact half.
  double boundary = 1e-16 / factor;
  while (boundary * factor > 1e-16) boundary = std::nextafter(boundary, 0.);
  double above = std::nextafter(boundary, 1.);
  while (above * factor <= 1e-16) above = std::nextafter(above, 1.);
  ASSERT_GT(boundary, 1e-16); // Input mask remains live on both sides.
  for (double amplitude : {std::nextafter(boundary, 0.), boundary, above, std::nextafter(above, 1.)}) {
    SCOPED_TRACE(amplitude);
    for (long b = 0; b < G.size(); ++b) G[b].data() = amplitude;
    auto actual = candidate.compute_self_energy(G, topology), expected = reference.compute_self_energy(G, topology);
    compare(actual, expected);
    EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
    if (amplitude <= boundary)
      EXPECT_EQ(norm(actual), 0.);
    else
      EXPECT_GT(candidate.last_contributing_paths(), 0);
  }
}
TEST(ScalarPathQualification, AdditionalWorkspaceAdmissionAndEmptyRange) {
  auto ad    = block_sparse_bench::atom(2, "occupation");
  auto coeff = block_sparse_bench::coefficients(ad);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G     = block_sparse_bench::propagator(ad, mesh);
  auto model = gpu::export_model(ad, coeff);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  gpu::ScalarPathEvaluator candidate(model, poles, mesh);
  gpu::DiagramEvaluator reference(model, poles, mesh);
  EXPECT_GT(candidate.self_energy_storage_bound(), reference.self_energy_storage_bound());
  gpu::ScalarPathEvaluator limited(model, poles, mesh, candidate.self_energy_storage_bound() - 1);
  EXPECT_THROW(limited.compute_self_energy_range(G, topology, 0, 1), std::length_error);
  EXPECT_EQ(limited.last_device_peak_delta(), 0);
  EXPECT_EQ(limited.last_path_counts().accepted, 0);
  EXPECT_EQ(norm(limited.compute_self_energy_range(G, topology, 0, 0)), 0.);
  compare(candidate.compute_self_energy(G, topology), reference.compute_self_energy(G, topology));
}
TEST(ScalarPathQualification, MultipleFieldScalarSetRetainsGeneralRoute) {
  auto ad    = block_sparse_bench::atom(2, "occupation");
  auto coeff = block_sparse_bench::coefficients(ad);
  using triqs::operators::n;
  std::vector<triqs::operators::many_body_operator_complex> operators{n("f", 0) + 1e-8, n("f", 1) + 1e-8};
  auto dyn     = nda::zeros<nda::dcomplex>(2, 2, 2);
  dyn(0, 0, 0) = .03;
  dyn(1, 1, 1) = .02;
  auto model   = gpu::export_model(ad, coeff, operators, dyn);
  ASSERT_TRUE(std::any_of(model.sets().begin(), model.sets().end(), [](auto const &set) { return set.size() > 1; }));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  gpu::DiagramEvaluator reference(model, poles, mesh);
  gpu::ScalarPathEvaluator candidate(model, poles, mesh);
  EXPECT_EQ(candidate.self_energy_storage_bound(), reference.self_energy_storage_bound());
  compare(candidate.compute_self_energy(G, topology), reference.compute_self_energy(G, topology));
}

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "scalar_evaluator.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <gtest/gtest.h>
#include "precision.hpp"
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
  gpu::ScalarPathEvaluator limited(model, poles, mesh, candidate.self_energy_storage_bound(topology) - 1);
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
#ifdef XCA_EXPERIMENT_SCALAR_SETS
  EXPECT_GT(candidate.self_energy_storage_bound(), reference.self_energy_storage_bound());
#else
  EXPECT_EQ(candidate.self_energy_storage_bound(), reference.self_energy_storage_bound());
#endif
  compare(candidate.compute_self_energy(G, topology), reference.compute_self_energy(G, topology));
}

TEST(ScalarPathQualification, ActiveOrderSixCutoffAndReallocatedInputs) {
  auto ad = block_sparse_bench::atom(2, "occupation");
  auto coeff = nda::zeros<nda::dcomplex>(1, 2, 2);
  coeff(0, 0, 0) = 1.;
  coeff(0, 1, 1) = 1.;
  auto model = gpu::export_model(ad, coeff);
  nda::vector<double> poles{0.};
  block_sparse_bench::CallOptions options;
  options.order = 6;
  options.topology = "chain";
  auto topology = block_sparse_bench::topology(options);
  triqs_xca::backbone::Backbone backbone(topology, model.num_fields(), model.num_interactions());
  long first = -1;
  // Bounded metadata-only discovery; assert activity instead of assuming first=1.
  for (long index = 0; index < std::min(4096L, backbone.num_flat_indices(poles.size())); ++index) {
    auto counts = gpu::for_each_self_energy_path(model, poles, topology, std::vector<bool>(ad.n_subspaces(), true),
                                                index, index + 1, [](auto &, auto const &) {});
    if (counts.accepted != 0) {
      first = index;
      break;
    }
  }
  ASSERT_GE(first, 0);
  SCOPED_TRACE(first);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 10., 1e-10, false);
  gpu::DiagramEvaluator reference(model, poles, mesh);
  gpu::ScalarPathEvaluator candidate(model, poles, mesh);
  auto G = block_sparse_bench::propagator(ad, mesh);
  long double reference_relative = 0, candidate_relative = 0;
  auto check_extended = [&](auto const &actual, auto const &expected, auto const &inputs) {
    auto precise = scalar_path_experiment::extended_self_energy(model, inputs, poles, topology, first, first + 1);
    long double signal = 0, reference_error = 0, candidate_error = 0;
    for (long b = 0; b < actual.size(); ++b)
      for (long t = 0; t < mesh.dlr_it().rank(); ++t) {
        auto z = precise[b](t, 0, 0);
        signal = std::max(signal, std::abs(z));
        reference_error = std::max(reference_error, std::abs(triqs_xca::test_utils::precision::extended(expected[b].data()(t, 0, 0)) - z));
        candidate_error = std::max(candidate_error, std::abs(triqs_xca::test_utils::precision::extended(actual[b].data()(t, 0, 0)) - z));
      }
    if (signal != 0) {
      reference_relative = std::max(reference_relative, reference_error / signal);
      candidate_relative = std::max(candidate_relative, candidate_error / signal);
    }
    // Order-six arithmetic accumulates >1e-13 relative error in both routes.
    // Keep the older comparisons unchanged; qualify this new case independently.
    EXPECT_LE(reference_error, 1e-12L * signal + 1e-30L);
    EXPECT_LE(candidate_error, 1e-12L * signal + 1e-30L);
  };
  auto evaluate = [&](double amplitude) {
    for (long b = 0; b < G.size(); ++b) G[b].data() = amplitude;
    auto expected = reference.compute_self_energy_range(G, topology, first, first + 1);
    auto actual = candidate.compute_self_energy_range(G, topology, first, first + 1);
    check_extended(actual, expected, G);
    EXPECT_GT(candidate.last_path_counts().accepted, 0);
    EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
    return reference.last_contributing_paths();
  };
  ASSERT_GT(evaluate(1.), 0);
  ASSERT_EQ(evaluate(1e-4), 0);
  // Locate the completed-path threshold using the real high-order arithmetic.
  // Probe both sides with a stated margin, rather than claiming an exact boundary.
  double low = 1e-4, high = 1.;
  for (int step = 0; step < 24; ++step) {
    double middle = (low + high) / 2;
    if (evaluate(middle) == 0) low = middle;
    else high = middle;
  }
  EXPECT_EQ(evaluate(low * (1. - 1e-4)), 0);
  EXPECT_GT(evaluate(high * (1. + 1e-4)), 0);
  // Keep old storage alive while supplying separately allocated, changed inputs.
  auto changed = block_sparse_bench::propagator(ad, mesh);
  for (long b = 0; b < changed.size(); ++b) changed[b].data() *= nda::dcomplex{.8, .1};
  auto actual = candidate.compute_self_energy_range(changed, topology, first, first + 1);
  auto expected = reference.compute_self_energy_range(changed, topology, first, first + 1);
  check_extended(actual, expected, changed);
  EXPECT_GT(candidate.last_contributing_paths(), 0);
  EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
  // Reusing the previous owner must not read stale data from the new allocation.
  EXPECT_GT(evaluate(1.), 0);
  std::cout << "Order-six relative error against long-double arithmetic: reference=" << reference_relative
            << " candidate=" << candidate_relative << " active_index=" << first << '\n';
}

#ifdef XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY
TEST(ScalarPathQualification, CutoffReplayAcrossCohortsAndPoleChoices) {
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  for (auto poles : {nda::vector<double>{0.}, nda::vector<double>{-1.5, .6}}) {
    SCOPED_TRACE(poles.size());
    auto coefficients = [&](int modes) {
      auto coeff = nda::zeros<nda::dcomplex>(poles.size(), modes, modes);
      for (long p = 0; p < poles.size(); ++p)
        for (int i = 0; i < modes; ++i) coeff(p, i, i) = 1.;
      return coeff;
    };
    auto small_ad = block_sparse_bench::atom(2, "occupation");
    auto small_model = gpu::export_model(small_ad, coefficients(2));
    auto small_G = block_sparse_bench::propagator(small_ad, mesh);
    for (long b = 0; b < small_G.size(); ++b) small_G[b].data() = 1.;
    gpu::DiagramEvaluator calibration(small_model, poles, mesh, 1024ULL * 1024 * 1024,
                                    32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
    auto unit = calibration.compute_self_energy_range(small_G, topology, 1, 2);
    ASSERT_EQ(calibration.last_contributing_paths(), 1);
    ASSERT_GT(norm(unit), 1e-16);
    // Three G factors per order-two lane. Calibration uses one isolated path.
    double center = std::cbrt(1e-16 / norm(unit));
    auto ad = block_sparse_bench::atom(6, "occupation");
    auto model = gpu::export_model(ad, coefficients(6));
    auto G = block_sparse_bench::propagator(ad, mesh);
    gpu::DiagramEvaluator reference(model, poles, mesh, 1024ULL * 1024 * 1024,
                                  32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
    gpu::ScalarPathEvaluator candidate(model, poles, mesh);
    for (double amplitude : {center * (1. - 1e-8), center * (1. - 1e-12), std::nextafter(center, 0.), center,
                             std::nextafter(center, 1.), center * (1. + 1e-12), center * (1. + 1e-8)}) {
      SCOPED_TRACE(amplitude);
      for (long b = 0; b < G.size(); ++b) G[b].data() = amplitude;
      auto expected = reference.compute_self_energy_range(G, topology, 1, 9);
      auto actual = candidate.compute_self_energy_range(G, topology, 1, 9);
      ASSERT_GT(candidate.last_path_counts().accepted, 512);
      ASSERT_NE(candidate.last_path_counts().accepted % 512, 0);
      EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
      compare(actual, expected);
      auto replay_count = candidate.last_scalar_replays();
      if (amplitude == center) {
        EXPECT_GT(replay_count, 0);
        if (poles.size() == 1) EXPECT_GT(replay_count, 512);
      }
      auto repeated = candidate.compute_self_energy_range(G, topology, 1, 9);
      EXPECT_EQ(candidate.last_scalar_replays(), replay_count);
      for (long b = 0; b < G.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(repeated[b].data() - actual[b].data())), 0.);
    }
  }
}

TEST(ScalarPathQualification, AdjacentOrderSixCutoffMatchesReference) {
  auto ad = block_sparse_bench::atom(2, "occupation");
  auto coeff = nda::zeros<nda::dcomplex>(1, 2, 2);
  coeff(0, 0, 0) = 1.;
  coeff(0, 1, 1) = 1.;
  auto model = gpu::export_model(ad, coeff);
  nda::vector<double> poles{0.};
  block_sparse_bench::CallOptions options;
  options.order = 6;
  options.topology = "chain";
  auto topology = block_sparse_bench::topology(options);
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 10., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  gpu::DiagramEvaluator reference(model, poles, mesh);
  gpu::ScalarPathEvaluator candidate(model, poles, mesh);
  constexpr long first = 10; // Active index independently discovered by the prior control.
  auto prepare = [&](double amplitude) { for (long b = 0; b < G.size(); ++b) G[b].data() = amplitude; };
  auto contributes = [&](double amplitude) {
    prepare(amplitude);
    reference.compute_self_energy_range(G, topology, first, first + 1);
    return reference.last_contributing_paths();
  };
  double low = 1e-4, high = 1.;
  ASSERT_EQ(contributes(low), 0);
  ASSERT_GT(contributes(high), 0);
  for (int step = 0; step < 64; ++step) {
    double middle = low + (high - low) / 2;
    if (middle == low || middle == high) break;
    if (contributes(middle) == 0) low = middle;
    else high = middle;
  }
  ASSERT_EQ(std::nextafter(low, high), high);
  for (double center : {low, high}) {
    double amplitude = center;
    for (int i = 0; i < 8; ++i) amplitude = std::nextafter(amplitude, 0.);
    for (int i = 0; i < 17; ++i, amplitude = std::nextafter(amplitude, 1.)) {
      SCOPED_TRACE(amplitude);
      prepare(amplitude);
      auto expected = reference.compute_self_energy_range(G, topology, first, first + 1);
      auto actual = candidate.compute_self_energy_range(G, topology, first, first + 1);
      ASSERT_GT(reference.last_path_counts().accepted, 0);
      EXPECT_EQ(candidate.last_contributing_paths(), reference.last_contributing_paths());
      double error = 0;
      for (long b = 0; b < G.size(); ++b) error = std::max(error, nda::max_element(nda::abs(actual[b].data() - expected[b].data())));
      EXPECT_LE(error, 1e-12 * std::max(norm(actual), norm(expected)) + 1e-30);
    }
  }
}
#endif

#include "test_utils/evaluators.hpp"
#include "triqs_xca/dense/diagram_evaluator_few_orb.hpp"
#include <gtest/gtest.h>

namespace {
  using namespace triqs_xca;
  using nda::dcomplex;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::n;
  using Tensor       = nda::array<dcomplex, 3>;
  constexpr auto all = nda::range::all;

  test_utils::EvalModel model(int modes = 2, bool high_rank = false) {
    triqs::atom_diag::fundamental_operator_set fops;
    triqs::operators::many_body_operator_complex H;
    test_utils::ops_t densities;
    std::vector<double> strengths;
    for (int i = 0; i < modes; ++i) {
      fops.insert("f", i);
      H += (-0.3 + 0.5 * i) * n("f", i);
      densities.push_back(n("f", i));
      strengths.push_back(0.7 - 0.1 * i);
      if (i > 0) {
        auto hopping = dcomplex(0.13, 0.07) * c_dag("f", i - 1) * c("f", i);
        H += 0.4 * n("f", i - 1) * n("f", i) + hopping + dagger(hopping);
      }
    }
    test_utils::ad_t ad(H, fops, test_utils::ops_t{});
    auto mixing = nda::zeros<dcomplex>(modes, modes);
    for (int i = 1; i < modes; ++i) {
      mixing(i - 1, i) = dcomplex(0.13, -0.04);
      mixing(i, i - 1) = dcomplex(-0.07, 0.02);
    }
    auto result = test_utils::make_eval_model("few_orb", ad, densities, test_utils::dynint_coeffs_from(strengths, mixing));
    if (high_rank) {
      constexpr double beta = 2., lambda = 1e5, eps = 1e-12;
      result.mesh = test_utils::mesh_t(beta, triqs::mesh::Fermion, lambda / beta, eps, false);
      auto ops    = result.mesh.dlr_it();
      auto atomic = test_utils::ad_to_atom_prop(result.ad, beta, ops);
      std::vector<triqs::gfs::gf<test_utils::mesh_t>> blocks;
      for (int b = 0; b < result.ad.n_subspaces(); ++b) blocks.emplace_back(result.mesh, atomic.get_block(b));
      result.G = test_utils::dressed_propagator(result.ad, result.mesh, test_utils::bgf_t(blocks));
    }
    return result;
  }

  template <diagram::Kind K>
  std::vector<diagram::Backbone<K>> prepared(Action<dense::FSet> const &action, long samples = 13, bool high_order = false) {
    std::vector<diagram::Backbone<K>> result;
    std::vector<nda::array<int, 2>> topologies{{{0, 3}, {1, 4}, {2, 5}}, {{0, 1}}, {{0, 1}, {2, 3}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 2}}};
    if (high_order) topologies = {{{0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}}};
    for (auto const &topology : topologies) {
      long count = diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), K);
      long size  = std::min(samples, count);
      for (long i = 0; i < size; ++i) result.push_back(diagram::prepare_backbone<K>(topology, action, i * (count - 1) / std::max(1L, size - 1)));
    }
    // Repetition guarantees reusable recipes without assuming a particular
    // traversal, replacement policy or internal cache-key representation.
    auto copy = result;
    result.insert(result.end(), copy.begin(), copy.end());
    return result;
  }

  void expect_close(Tensor const &actual, Tensor const &expected) {
    ASSERT_EQ(actual.shape(), expected.shape());
    if (actual.size() == 0) return;
    double scale = std::max(1., nda::max_element(nda::abs(expected)));
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 2e-11 * scale);
  }

  std::size_t entry_bytes(Action<dense::FSet> const &action) {
    return 2 * action.tau_mesh.size() * action.block_dims(0) * action.block_dims(0) * sizeof(dcomplex);
  }

  template <typename Compute>
  Tensor sum_self_energy(dense::DiagramEvaluator &scalar, Tensor const &G, std::vector<diagram::SelfEnergyBackbone> const &backbones,
                         Compute compute) {
    auto result = nda::zeros<dcomplex>(G.shape());
    for (auto const &backbone : backbones) result += compute(scalar, G, backbone);
    return result;
  }
} // namespace

TEST(DenseFewOrb, self_energy_pairs_cache_limits_and_propagator_changes) {
  auto m      = model();
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  dense::DiagramEvaluator scalar(action);
  auto backbones = prepared<diagram::Kind::self_energy>(action);
  auto ordinary  = [](auto &e, auto const &G, auto const &b) { return e.compute_self_energy(G, b); };
  auto paired    = [](auto &e, auto const &G, auto const &b) { return e.compute_self_energy_by_pairs(G, b); };
  auto expected  = sum_self_energy(scalar, m.G, backbones, ordinary);
  auto pairs     = sum_self_energy(scalar, m.G, backbones, paired);
  auto G_changed = test_utils::second_propagator(m);
  auto changed   = sum_self_energy(scalar, G_changed, backbones, ordinary);
  ASSERT_GT(nda::max_element(nda::abs(expected)), 1e-6);
  ASSERT_GT(nda::max_element(nda::abs(changed - expected)), 1e-6);

  std::size_t bytes = entry_bytes(action);
  for (std::size_t budget : {std::size_t{0}, bytes - 1, bytes, 16 * bytes, 1024 * bytes}) {
    SCOPED_TRACE(budget);
    dense::DiagramEvaluatorFewOrb evaluator(action, budget);
    Tensor G      = m.G;
    auto *address = G.data();
    auto retained = evaluator.compute_self_energy(G, backbones);
    expect_close(retained, expected);
    auto stats = evaluator.edge_cache_statistics();
    EXPECT_LE(stats.bytes, budget);
    EXPECT_EQ(stats.bytes, stats.entries * bytes);
    EXPECT_GT(stats.misses, 0);
    if (budget < bytes) {
      EXPECT_EQ(stats.entries, 0);
      EXPECT_EQ(stats.hits, 0);
    } else {
      EXPECT_GT(stats.hits, 0);
      EXPECT_GT(stats.entries, 0);
    }
    if (budget == bytes) {
      EXPECT_EQ(stats.entries, 1);
      EXPECT_GT(stats.misses, stats.entries);
    }
    expect_close(evaluator.compute_self_energy_by_pairs(G, backbones), pairs);
    // Keep the same allocation: pointer identity cannot establish cache validity.
    G() = G_changed;
    ASSERT_EQ(G.data(), address);
    expect_close(evaluator.compute_self_energy(G, backbones), changed);
    G() = m.G;
    ASSERT_EQ(G.data(), address);
    expect_close(evaluator.compute_self_energy(G, backbones), expected);
    auto repeated_stats = evaluator.edge_cache_statistics();
    EXPECT_EQ(repeated_stats.hits, stats.hits);
    EXPECT_EQ(repeated_stats.misses, stats.misses);
    expect_close(retained, expected);
    expect_close(evaluator.compute_self_energy(m.G, backbones.front()), scalar.compute_self_energy(m.G, backbones.front()));
    expect_close(evaluator.compute_self_energy_by_pairs(m.G, backbones.back()), scalar.compute_self_energy_by_pairs(m.G, backbones.back()));
  }
}

TEST(DenseFewOrb, correlator_reflection_statistics_and_cache_limits) {
  auto m      = model();
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  dense::DiagramEvaluator scalar(action);
  auto backbones    = prepared<diagram::Kind::correlator>(action);
  auto annihilators = action.operators.Fs(nda::range(action.n_hyb), all, all);
  auto creators     = action.operators.F_dags(nda::range(action.n_hyb), all, all);
  auto densities    = action.operators.Fs(nda::range(action.n_hyb, action.n_flavours()), all, all);
  std::size_t bytes = entry_bytes(action);
  for (bool bosonic : {false, true}) {
    auto mu         = bosonic ? densities : annihilators;
    auto kap        = bosonic ? densities : creators;
    auto statistics = diagram::uniform_correlator_statistics(!bosonic);
    auto expected   = nda::zeros<dcomplex>(m.mesh.size(), mu.extent(0), kap.extent(0));
    for (auto const &b : backbones) expected += scalar.compute_one_time_correlator(m.G, b, mu, kap, statistics);
    ASSERT_GT(nda::max_element(nda::abs(expected)), 1e-6);
    for (std::size_t budget : {std::size_t{0}, bytes, 1024 * bytes}) {
      SCOPED_TRACE(budget);
      dense::DiagramEvaluatorFewOrb evaluator(action, budget);
      auto retained = evaluator.compute_one_time_correlator(m.G, backbones, mu, kap, statistics);
      expect_close(retained, expected);
      auto stats = evaluator.edge_cache_statistics();
      EXPECT_LE(stats.bytes, budget);
      expect_close(evaluator.compute_one_time_correlator(m.G, backbones.front(), mu, kap, statistics),
                   scalar.compute_one_time_correlator(m.G, backbones.front(), mu, kap, statistics));
      expect_close(evaluator.compute_one_time_correlator(m.G, backbones, mu, kap, statistics), expected);
      expect_close(retained, expected);
    }
  }
}

TEST(DenseFewOrb, fixed_dimensions_blas_fallback_and_high_rank) {
  for (auto [modes, high_rank] : {std::pair{1, false}, std::pair{2, false}, std::pair{3, false}, std::pair{4, false},
                                 std::pair{5, false}, std::pair{2, true}}) {
    SCOPED_TRACE(modes);
    SCOPED_TRACE(high_rank);
    auto m      = model(modes, high_rank);
    auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
    ASSERT_EQ(action.block_dims(0), 1 << modes);
    if (high_rank) ASSERT_GE(m.mesh.size(), 100);
    dense::DiagramEvaluator scalar(action);
    // Keep the existing high-order checks for N=2,8 and r~=100. The larger
    // dimensions use a bounded order-two sample, including N=32 BLAS fallback.
    auto sample = [&]<diagram::Kind K>() {
      if (high_rank || modes == 1 || modes == 3) return prepared<K>(action, 5, true);
      nda::array<int, 2> topology{{0, 2}, {1, 3}};
      long count = diagram::count_diagrams(2, action.n_flavours(), action.poles.size(), K);
      std::vector<diagram::Backbone<K>> result;
      for (long i : {0L, count / 3, 2 * count / 3, count - 1}) result.push_back(diagram::prepare_backbone<K>(topology, action, i));
      auto copy = result;
      result.insert(result.end(), copy.begin(), copy.end());
      return result;
    };
    auto backbones   = sample.template operator()<diagram::Kind::self_energy>();
    auto correlators = sample.template operator()<diagram::Kind::correlator>();
    auto mu          = action.operators.Fs(nda::range(action.n_hyb), all, all);
    auto kap         = action.operators.F_dags(nda::range(action.n_hyb), all, all);
    auto expected = nda::zeros<dcomplex>(m.G.shape()), pairs = nda::zeros<dcomplex>(m.G.shape());
    auto corr = nda::zeros<dcomplex>(m.mesh.size(), mu.extent(0), kap.extent(0));
    for (auto const &b : backbones) {
      expected += scalar.compute_self_energy(m.G, b);
      pairs += scalar.compute_self_energy_by_pairs(m.G, b);
    }
    for (auto const &b : correlators) corr += scalar.compute_one_time_correlator(m.G, b, mu, kap);
    for (std::size_t budget : {std::size_t{0}, 256 * entry_bytes(action)}) {
      dense::DiagramEvaluatorFewOrb evaluator(action, budget);
      expect_close(evaluator.compute_self_energy(m.G, backbones), expected);
      expect_close(evaluator.compute_self_energy_by_pairs(m.G, backbones), pairs);
      expect_close(evaluator.compute_one_time_correlator(m.G, correlators, mu, kap), corr);
    }
  }
}

TEST(DenseFewOrb, topology_empty_input_validation_and_recovery) {
  auto m      = model();
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  dense::DiagramEvaluator scalar(action);
  dense::DiagramEvaluatorFewOrb evaluator(action);
  test_utils::bgf_t G{std::vector{triqs::gfs::gf<test_utils::mesh_t>(m.mesh, m.G)}};
  nda::array<int, 2> topology{{0, 2}, {1, 3}}, malformed{{0, 0}};
  auto mu  = action.operators.Fs(nda::range(action.n_hyb), all, all);
  auto kap = action.operators.F_dags(nda::range(action.n_hyb), all, all);
  expect_close(evaluator.compute_self_energy(G, topology)[0].data(), scalar.compute_self_energy(G, topology)[0].data());
  expect_close(evaluator.compute_self_energy_by_pairs(G, topology)[0].data(), scalar.compute_self_energy_by_pairs(G, topology)[0].data());
  expect_close(evaluator.compute_single_ptcle_gf(G, topology), scalar.compute_single_ptcle_gf(G, topology));
  test_utils::ops_t ops{n("f", 0), n("f", 1)};
  expect_close(evaluator.compute_one_time_correlator(G, ops, ops, m.ad_flat, topology),
               scalar.compute_one_time_correlator(G, ops, ops, m.ad_flat, topology));

  std::span<diagram::SelfEnergyBackbone const> empty;
  std::span<diagram::CorrelatorBackbone const> empty_correlator;
  EXPECT_EQ(nda::max_element(nda::abs(evaluator.compute_self_energy(m.G, empty))), 0.);
  EXPECT_EQ(nda::max_element(nda::abs(evaluator.compute_self_energy_by_pairs(m.G, empty))), 0.);
  EXPECT_EQ(nda::max_element(nda::abs(evaluator.compute_one_time_correlator(m.G, empty_correlator, mu, kap))), 0.);
  EXPECT_EQ(evaluator.edge_cache_statistics().entries, 0);
  EXPECT_EQ(evaluator.edge_cache_statistics().hits, 0);
  EXPECT_EQ(evaluator.edge_cache_statistics().misses, 0);
  auto wrong_shape = nda::zeros<dcomplex>(m.mesh.size(), 3, 3);
  auto wrong_ops   = nda::zeros<dcomplex>(2, 3, 3);
  EXPECT_THROW(evaluator.compute_self_energy(wrong_shape, empty), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_self_energy_by_pairs(wrong_shape, empty), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_one_time_correlator(m.G, empty_correlator, wrong_ops, kap), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_self_energy(G, malformed), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_self_energy_by_pairs(G, malformed), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_single_ptcle_gf(G, malformed), std::invalid_argument);
  test_utils::bgf_t missing{std::vector<triqs::gfs::gf<test_utils::mesh_t>>{}};
  EXPECT_THROW(evaluator.compute_self_energy(missing, topology), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_one_time_correlator(missing, topology, mu, kap), std::invalid_argument);
  expect_close(evaluator.compute_self_energy(G, topology)[0].data(), scalar.compute_self_energy(G, topology)[0].data());
}

TEST(DenseFewOrb, strided_propagator_and_observable_inputs) {
  auto m      = model();
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  dense::DiagramEvaluator scalar(action);
  auto backbones   = prepared<diagram::Kind::self_energy>(action, 3);
  auto correlators = prepared<diagram::Kind::correlator>(action, 3);
  auto mu          = action.operators.Fs(nda::range(action.n_hyb), all, all);
  auto kap         = action.operators.F_dags(nda::range(action.n_hyb), all, all);
  long r = m.mesh.size(), N = action.block_dims(0);
  Tensor G_store(2 * r, 2 * N, 2 * N), mu_store(2 * action.n_hyb, 2 * N, 2 * N), kap_store(mu_store.shape());
  G_store() = dcomplex{13., -17.};
  mu_store() = dcomplex{13., -17.};
  kap_store() = dcomplex{13., -17.};
  auto space                           = nda::range(0, 2 * N, 2);
  auto operators                       = nda::range(0, 2 * action.n_hyb, 2);
  auto G_strided                       = G_store(nda::range(0, 2 * r, 2), space, space);
  auto mu_strided                      = mu_store(operators, space, space);
  auto kap_strided                     = kap_store(operators, space, space);
  G_strided                            = m.G;
  mu_strided                           = mu;
  kap_strided                          = kap;
  ASSERT_FALSE(G_strided.indexmap().is_contiguous());
  ASSERT_FALSE(mu_strided.indexmap().is_contiguous());
  auto expected = nda::zeros<dcomplex>(m.G.shape()), pairs = nda::zeros<dcomplex>(m.G.shape());
  auto corr = nda::zeros<dcomplex>(r, mu.extent(0), kap.extent(0));
  for (auto const &b : backbones) {
    expected += scalar.compute_self_energy(m.G, b);
    pairs += scalar.compute_self_energy_by_pairs(m.G, b);
  }
  for (auto const &b : correlators) corr += scalar.compute_one_time_correlator(m.G, b, mu, kap);
  for (std::size_t budget : {std::size_t{0}, 64 * entry_bytes(action)}) {
    dense::DiagramEvaluatorFewOrb evaluator(action, budget);
    expect_close(evaluator.compute_self_energy(G_strided, backbones), expected);
    expect_close(evaluator.compute_self_energy_by_pairs(G_strided, backbones), pairs);
    expect_close(evaluator.compute_one_time_correlator(G_strided, correlators, mu_strided, kap_strided), corr);
  }
  EXPECT_EQ(G_store(2 * r - 1, 2 * N - 1, 2 * N - 1), (dcomplex{13., -17.}));
  EXPECT_EQ(mu_store(2 * action.n_hyb - 1, 2 * N - 1, 2 * N - 1), (dcomplex{13., -17.}));
  EXPECT_EQ(kap_store(2 * action.n_hyb - 1, 2 * N - 1, 2 * N - 1), (dcomplex{13., -17.}));
}

#include "test_utils/evaluators.hpp"
#include "triqs_xca/dense/diagram_evaluator_batched.hpp"
#include <gtest/gtest.h>

namespace {
  using namespace triqs_xca;
  using nda::dcomplex;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::n;
  using Tensor       = nda::array<dcomplex, 3>;
  constexpr auto all = nda::range::all;

  test_utils::EvalModel model() {
    triqs::atom_diag::fundamental_operator_set fops;
    fops.insert("f", 0);
    fops.insert("f", 1);
    auto hopping = dcomplex(0.13, 0.07) * c_dag("f", 0) * c("f", 1);
    auto H       = -0.3 * n("f", 0) + 0.2 * n("f", 1) + 0.4 * n("f", 0) * n("f", 1) + hopping + dagger(hopping);
    test_utils::ad_t ad(H, fops, test_utils::ops_t{});
    nda::matrix<dcomplex> mixing{{0., dcomplex(0.13, -0.04)}, {dcomplex(-0.07, 0.02), 0.}};
    return test_utils::make_eval_model("batched_N4", ad, {n("f", 0), n("f", 1)}, test_utils::dynint_coeffs_from({0.7, 0.4}, mixing));
  }

  template <diagram::Kind K> std::vector<diagram::Backbone<K>> prepared(Action<dense::FSet> const &action) {
    std::vector<diagram::Backbone<K>> result;
    // Include both endpoint extremes, a crossing and two consecutive cohort
    // changes. None of these lengths divides the tested batch capacities.
    for (auto topology : {nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}, nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 1}, {2, 3}},
                          nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 3}, {1, 2}}}) {
      long count   = diagram::count_diagrams(topology.extent(0), action.n_flavours(), action.poles.size(), K);
      long samples = std::min(17L, count);
      for (long i = 0; i < samples; ++i)
        result.push_back(diagram::prepare_backbone<K>(topology, action, i * (count - 1) / std::max(1L, samples - 1)));
    }
    return result;
  }

  void expect_close(Tensor const &actual, Tensor const &expected) {
    ASSERT_EQ(actual.shape(), expected.shape());
    double scale = nda::max_element(nda::abs(expected));
    ASSERT_GT(scale, 1e-10) << "the fixture must resolve a nonzero result";
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 2e-11 * scale);
  }
} // namespace

TEST(DenseBatched, self_energy_pairs_tails_and_reuse) {
  auto m      = model();
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  dense::DiagramEvaluator scalar(action);
  auto backbones = prepared<diagram::Kind::self_energy>(action);
  auto expected = nda::zeros<dcomplex>(m.G.shape());
  auto changed_G = nda::make_regular(1.03 * m.G);
  auto changed   = nda::zeros<dcomplex>(m.G.shape());
  for (auto const &backbone : backbones) {
    expected += scalar.compute_self_energy_both_directions(m.G, backbone);
    changed += scalar.compute_self_energy_both_directions(changed_G, backbone);
  }
  for (long capacity : {1, 3, 8, 19}) {
    SCOPED_TRACE(capacity);
    dense::BatchedDiagramEvaluator evaluator(action, capacity);
    auto retained = evaluator.compute_self_energy_both_directions(m.G, backbones);
    expect_close(retained, expected);
    expect_close(evaluator.compute_self_energy_both_directions(changed_G, backbones), changed);
    expect_close(evaluator.compute_self_energy_both_directions(m.G, backbones), expected);
    expect_close(retained, expected);
  }
}

TEST(DenseBatched, correlator_reflection_statistics_and_tails) {
  auto m      = model();
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  dense::DiagramEvaluator scalar(action);
  auto backbones    = prepared<diagram::Kind::correlator>(action);
  auto annihilators = action.operators.Fs(nda::range(action.n_hyb), all, all);
  auto creators     = action.operators.F_dags(nda::range(action.n_hyb), all, all);
  auto densities    = action.operators.Fs(nda::range(action.n_hyb, action.n_flavours()), all, all);
  for (bool bosonic : {false, true}) {
    auto mu         = bosonic ? densities : annihilators;
    auto kap        = bosonic ? densities : creators;
    auto statistics = diagram::uniform_correlator_statistics(!bosonic);
    auto expected   = nda::zeros<dcomplex>(m.mesh.size(), mu.extent(0), kap.extent(0));
    for (auto const &backbone : backbones) expected += scalar.compute_one_time_correlator(m.G, backbone, mu, kap, statistics);
    for (long capacity : {1, 3, 8, 19}) {
      SCOPED_TRACE(capacity);
      dense::BatchedDiagramEvaluator evaluator(action, capacity);
      expect_close(evaluator.compute_one_time_correlator(m.G, backbones, mu, kap, statistics), expected);
      expect_close(evaluator.compute_one_time_correlator(m.G, backbones, mu, kap, statistics), expected);
    }
  }
}

TEST(DenseBatched, topology_wrappers_and_empty_input) {
  auto m      = model();
  auto action = dense::make_action(m.mesh, m.hyb_poles, m.hyb_coeffs, m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  dense::DiagramEvaluator scalar(action);
  dense::BatchedDiagramEvaluator evaluator(action, 3);
  test_utils::bgf_t G{std::vector{triqs::gfs::gf<test_utils::mesh_t>(m.mesh, m.G)}};
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  expect_close(evaluator.compute_self_energy(G, topology)[0].data(), scalar.compute_self_energy(G, topology)[0].data());
  expect_close(evaluator.compute_single_ptcle_gf(G, topology), scalar.compute_single_ptcle_gf(G, topology));
  std::span<diagram::SelfEnergyBackbone const> empty;
  EXPECT_EQ(nda::max_element(nda::abs(evaluator.compute_self_energy_both_directions(m.G, empty))), 0.);
  EXPECT_THROW((dense::BatchedDiagramEvaluator(action, 0)), std::invalid_argument);
  auto wrong_shape = nda::zeros<dcomplex>(m.mesh.size(), 3, 3);
  EXPECT_THROW(evaluator.compute_self_energy_both_directions(wrong_shape, empty), std::invalid_argument);
}

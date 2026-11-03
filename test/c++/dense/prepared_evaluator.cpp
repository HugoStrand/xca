#include "test_utils/prepared_evaluators.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include <gtest/gtest.h>

#include <triqs_xca/dense/diagram_evaluator.hpp>
#include "test_utils/evaluators.hpp"

using triqs_xca::diagram::Direction;

namespace {
using namespace triqs_xca;
using nda::dcomplex;
using tensor = nda::array<dcomplex, 3>;

// A wrong endpoint operation must be rejected before evaluation can change scratch or results.
template <typename B> concept SelfEnergyInput = requires(dense::DiagramEvaluator &e, tensor const &G, B const &b) {
  e.eval_self_energy(G, b);
};
template <typename B> concept SelfEnergyPairInput = requires(dense::DiagramEvaluator &e, tensor const &G, B const &b) {
  e.eval_self_energy_pair(G, b);
};
template <typename B> concept CorrelatorInput = requires(dense::DiagramEvaluator &e, tensor const &G, B const &b) {
  e.eval_correlator(G, b, G, G);
};
static_assert(SelfEnergyInput<diagram::SelfEnergyBackbone> && !SelfEnergyInput<diagram::CorrelatorBackbone>);
static_assert(SelfEnergyPairInput<diagram::SelfEnergyBackbone> && !SelfEnergyPairInput<diagram::CorrelatorBackbone>);
static_assert(CorrelatorInput<diagram::CorrelatorBackbone> && !CorrelatorInput<diagram::SelfEnergyBackbone>);

void expect_close(tensor const &actual, tensor const &expected) {
  double scale = nda::max_element(nda::abs(expected));
  ASSERT_GT(scale, 1e-14) << "the selected contribution must be resolved";
  EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 1e-12 * scale);
}
}

TEST(DensePrepared, self_energy_and_pair_outlive_preparation_inputs) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  dense::DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  auto G = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>{std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(model.mesh, model.G)}};
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  int const n = evaluator.action.n_flavours();
  auto const backbone = [&] {
    nda::array<int, 2> preparation_topology{topology};
    auto arcs = triqs_xca::diagram::prepare_arcs(
      triqs_xca::diagram::Arcs{{Direction::backward, -1, -1}, {Direction::forward, n - 1, 1}},
      evaluator.action, triqs_xca::diagram::Kind::self_energy);
    auto result = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(preparation_topology, arcs);
    preparation_topology() = 0;
    arcs.clear();
    return result;
  }();
  long f = (n - 1) + n * (1 + model.hyb_poles.size() * 2);
  long partner = f + n * model.hyb_poles.size();
  auto expected = tensor(triqs_xca::test_utils::fixture_self_energy(evaluator, G, topology, f)[0].data());
  auto expected_pair = nda::make_regular(expected + triqs_xca::test_utils::fixture_self_energy(evaluator, G, topology, partner)[0].data());

  evaluator.reset();
  evaluator.eval_self_energy(model.G, backbone);
  expect_close(evaluator.Sigma, expected);
  evaluator.reset();
  evaluator.eval_self_energy_pair(model.G, backbone);
  expect_close(evaluator.Sigma, expected_pair);
  EXPECT_FALSE(backbone.endpoint_orientation().forward);
  EXPECT_TRUE(backbone.vertices[0].adjoint);
  auto copy = backbone;
  copy.vertices[1].flavour = 0;
  copy.endpoints.special_vertex = 1;
  EXPECT_EQ(backbone.vertices[1].flavour, n - 1);
  EXPECT_EQ(backbone.endpoints.special_vertex, 2);
  EXPECT_FALSE(backbone.endpoint_orientation().forward);

  // There are three propagator edges at order two. Reusing preparation with a
  // different G must not reuse old numerical data or an old accumulator.
  evaluator.reset();
  auto changed_G = nda::make_regular(1.1 * model.G);
  evaluator.eval_self_energy(changed_G, backbone);
  expect_close(evaluator.Sigma, nda::make_regular((1.1 * 1.1 * 1.1) * expected));
}

TEST(DensePrepared, correlator_outlives_preparation_inputs) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  dense::DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  auto G = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>{std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(model.mesh, model.G)}};
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  int const n = evaluator.action.n_flavours();
  auto const backbone = [&] {
    nda::array<int, 2> preparation_topology{topology};
    auto arcs = triqs_xca::diagram::prepare_arcs(
      triqs_xca::diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, n - 1, 1}},
      evaluator.action, triqs_xca::diagram::Kind::correlator);
    auto result = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::correlator>(preparation_topology, arcs);
    preparation_topology() = 0;
    arcs.clear();
    return result;
  }();
  long f = (n - 1) + n * (1 + model.hyb_poles.size());
  auto expected = triqs_xca::test_utils::fixture_single_ptcle_gf(evaluator, G, topology, f);
  auto annihilators = nda::make_regular(evaluator.action.operators.Fs(nda::range(evaluator.action.n_hyb), nda::range::all, nda::range::all));
  auto creators = nda::make_regular(evaluator.action.operators.F_dags(nda::range(evaluator.action.n_hyb), nda::range::all, nda::range::all));
  auto actual = evaluator.eval_correlator(model.G, backbone, annihilators, creators, triqs_xca::diagram::CorrelatorStatistics{});
  expect_close(actual, expected);
  auto changed_G = nda::make_regular(1.1 * model.G);
  auto changed = evaluator.eval_correlator(changed_G, backbone, annihilators, creators, triqs_xca::diagram::CorrelatorStatistics{});
  expect_close(changed, nda::make_regular((1.1 * 1.1 * 1.1 * 1.1) * expected));
  EXPECT_EQ(backbone.vertices[0].flavour, -1);
}

TEST(DensePrepared, self_energy_accumulates_prepared_contributions) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  dense::DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  nda::array<int, 2> topology{{0, 1}};
  auto sigma = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, evaluator.action, 0);
  evaluator.eval_self_energy(model.G, sigma);
  auto expected = evaluator.Sigma;
  evaluator.eval_self_energy(model.G, sigma);
  expect_close(evaluator.Sigma, nda::make_regular(2. * expected));
}

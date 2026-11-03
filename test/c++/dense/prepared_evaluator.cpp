#include "test_utils/prepared_evaluators.hpp"
#include "triqs_xca/diagram/backbone.hpp"
#include <concepts>
#include <memory>
#include <tuple>
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
  { e.compute_self_energy(G, b) } -> std::same_as<tensor>;
};
template <typename B> concept SelfEnergyPairInput = requires(dense::DiagramEvaluator &e, tensor const &G, B const &b) {
  { e.compute_self_energy_by_pairs(G, b) } -> std::same_as<tensor>;
};
template <typename B> concept CorrelatorInput = requires(dense::DiagramEvaluator &e, tensor const &G, B const &b) {
  { e.compute_one_time_correlator(G, b, G, G) } -> std::same_as<tensor>;
};
static_assert(SelfEnergyInput<diagram::SelfEnergyBackbone> && !SelfEnergyInput<diagram::CorrelatorBackbone>);
static_assert(SelfEnergyPairInput<diagram::SelfEnergyBackbone> && !SelfEnergyPairInput<diagram::CorrelatorBackbone>);
static_assert(CorrelatorInput<diagram::CorrelatorBackbone> && !CorrelatorInput<diagram::SelfEnergyBackbone>);
template <typename E> concept PublicAccumulator = requires(E &e) { e.Sigma; };
template <typename E> concept PublicReset = requires(E &e) { e.reset(); };
static_assert(!PublicAccumulator<dense::DiagramEvaluator> && !PublicReset<dense::DiagramEvaluator>);

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

  expect_close(evaluator.compute_self_energy(model.G, backbone), expected);
  expect_close(evaluator.compute_self_energy_by_pairs(model.G, backbone), expected_pair);
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
  auto changed_G = nda::make_regular(1.1 * model.G);
  expect_close(evaluator.compute_self_energy(changed_G, backbone), nda::make_regular((1.1 * 1.1 * 1.1) * expected));
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
  auto actual = evaluator.compute_one_time_correlator(model.G, backbone, annihilators, creators, triqs_xca::diagram::CorrelatorStatistics{});
  expect_close(actual, expected);
  auto changed_G = nda::make_regular(1.1 * model.G);
  auto changed = evaluator.compute_one_time_correlator(changed_G, backbone, annihilators, creators, triqs_xca::diagram::CorrelatorStatistics{});
  expect_close(changed, nda::make_regular((1.1 * 1.1 * 1.1 * 1.1) * expected));
  EXPECT_EQ(backbone.vertices[0].flavour, -1);
}

TEST(DensePrepared, repeated_calls_return_independent_contributions) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  auto fresh = [&] {
    return dense::DiagramEvaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  };
  auto evaluator = fresh();
  nda::array<int, 2> topology{{0, 1}};
  auto sigma = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, evaluator.action, 0);
  auto correlator = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::correlator>(topology, evaluator.action, 0);
  auto annihilators = evaluator.action.operators.Fs(nda::range(evaluator.action.n_hyb), nda::range::all, nda::range::all);
  auto creators = evaluator.action.operators.F_dags(nda::range(evaluator.action.n_hyb), nda::range::all, nda::range::all);

  // Independent evaluator instances provide references without any preceding call.
  auto expected_sigma = fresh().compute_self_energy(model.G, sigma);
  auto expected_pair = fresh().compute_self_energy_by_pairs(model.G, sigma);
  auto expected_correlator = fresh().compute_one_time_correlator(model.G, correlator, annihilators, creators);

  auto actual_sigma = evaluator.compute_self_energy(model.G, sigma);
  auto actual_pair = evaluator.compute_self_energy_by_pairs(model.G, sigma);
  auto actual_correlator = evaluator.compute_one_time_correlator(model.G, correlator, annihilators, creators);
  expect_close(actual_sigma, expected_sigma);
  expect_close(actual_pair, expected_pair);
  expect_close(actual_correlator, expected_correlator);

  // Reuse the three prepared operations with different data, without a reset.
  auto changed_G = nda::make_regular(1.1 * model.G);
  expect_close(evaluator.compute_self_energy(changed_G, sigma), nda::make_regular(1.1 * expected_sigma));
  expect_close(evaluator.compute_self_energy_by_pairs(changed_G, sigma), nda::make_regular(1.1 * expected_pair));
  expect_close(evaluator.compute_one_time_correlator(changed_G, correlator, annihilators, creators),
               nda::make_regular((1.1 * 1.1) * expected_correlator));

  expect_close(evaluator.compute_self_energy(model.G, sigma), expected_sigma);
  expect_close(evaluator.compute_self_energy_by_pairs(model.G, sigma), expected_pair);
  expect_close(evaluator.compute_one_time_correlator(model.G, correlator, annihilators, creators), expected_correlator);

  // Returned arrays own their storage and survive subsequent evaluator calls.
  expect_close(actual_sigma, expected_sigma);
  expect_close(actual_pair, expected_pair);
  expect_close(actual_correlator, expected_correlator);
}

TEST(DensePrepared, copied_evaluator_retains_convolution_grid) {
  // Destroy both the original evaluator and model before using the copy. The
  // Action mesh shares its interpolation object, which fast_convolve borrows.
  auto [retained, G, prepared, reference] = [] {
    auto model = test_utils::build_eval_model("unequal_sets_nA0");
    dense::DiagramEvaluator original(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
    nda::array<int, 2> topology{{0, 2}, {1, 3}};
    auto backbone = diagram::prepare_backbone<diagram::Kind::self_energy>(topology, original.action, 0);
    auto expected = original.compute_self_energy_by_pairs(model.G, backbone);
    auto copy = std::make_unique<dense::DiagramEvaluator>(original);
    EXPECT_EQ(&copy->action.tau_mesh.dlr_it(), &original.action.tau_mesh.dlr_it());
    return std::tuple{std::move(copy), model.G, std::move(backbone), std::move(expected)};
  }();
  expect_close(retained->compute_self_energy_by_pairs(G, prepared), reference);
}

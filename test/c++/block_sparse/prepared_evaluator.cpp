#include "test_utils/prepared_evaluators.hpp"
#include <triqs_xca/block_sparse/self_energy_paths.hpp>
#include "triqs_xca/diagram/backbone.hpp"
#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/dynint.hpp>

#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include "test_utils/evaluators.hpp"

using triqs_xca::diagram::Direction;

namespace {
using namespace triqs_xca;
using namespace triqs_xca::block_sparse;
using nda::dcomplex;
using tensor = nda::array<dcomplex, 3>;

void expect_close(tensor const &actual, tensor const &expected) {
  double scale = nda::max_element(nda::abs(expected));
  ASSERT_GT(scale, 1e-14) << "the selected contribution must be resolved";
  EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 1e-12 * scale);
}

BlockOp identity(BlockDiagOpFun const &G) {
  nda::vector<int> destinations(G.get_num_block_cols());
  std::vector<nda::array<dcomplex, 2>> blocks;
  for (int b = 0; b < G.get_num_block_cols(); ++b) {
    destinations(b) = b;
    int d = G.get_block_size(b);
    auto block = nda::zeros<dcomplex>(d, d);
    for (int i = 0; i < d; ++i) block(i, i) = 1;
    blocks.emplace_back(block);
  }
  return {destinations, blocks};
}
}

TEST(BlockSparsePrepared, logical_self_energy_reuses_preparation_with_changed_propagator) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
  BlockDiagOpFun G(test_utils::subspace_blocks(model.G, model.ad, model.mesh));
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  int n = evaluator.n;
  auto const assignment = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(
     topology,
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::backward, -1, -1}, {Direction::forward, n - 1, 1}}, evaluator.action,
                                      triqs_xca::diagram::Kind::self_energy));
  long f = (n - 1) + n * (1 + model.hyb_poles.size() * 2);
  auto expected = test_utils::get_tensor_in_full_hilbert_space(triqs_xca::test_utils::fixture_self_energy(evaluator, G, topology, f), model.ad);
  auto actual = test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(G, assignment), model.ad);
  expect_close(actual, expected);

  auto changed_G = G;
  changed_G *= 1.1;
  auto changed = test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(changed_G, assignment), model.ad);
  expect_close(changed, nda::make_regular((1.1 * 1.1 * 1.1) * expected));
  EXPECT_FALSE(assignment.endpoint_orientation().forward);
  EXPECT_TRUE(assignment.vertices[0].adjoint);
}

TEST(BlockSparsePrepared, logical_correlator_reuses_preparation_with_changed_propagator) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
  BlockDiagOpFun G(test_utils::subspace_blocks(model.G, model.ad, model.mesh));
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  int n = evaluator.n;
  auto const assignment = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::correlator>(
     topology,
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, n - 1, 1}}, evaluator.action,
                                      triqs_xca::diagram::Kind::correlator));
  long f = (n - 1) + n * (1 + model.hyb_poles.size());
  auto op = identity(G);
  auto expected = triqs_xca::test_utils::fixture_correlator(evaluator, G, topology, std::vector<BlockOp>{op}, std::vector<BlockOp>{op}, f, false);
  auto actual = evaluator.eval_correlator(G, assignment, {op}, {op}, triqs_xca::diagram::CorrelatorStatistics{triqs_xca::Statistics::bosonic, triqs_xca::Statistics::bosonic});
  expect_close(actual, expected);

  auto changed_G = G;
  changed_G *= 1.1;
  auto changed = evaluator.eval_correlator(changed_G, assignment, {op}, {op}, triqs_xca::diagram::CorrelatorStatistics{triqs_xca::Statistics::bosonic, triqs_xca::Statistics::bosonic});
  expect_close(changed, nda::make_regular((1.1 * 1.1 * 1.1 * 1.1) * expected));
  EXPECT_EQ(assignment.vertices[0].flavour, -1);
}


TEST(BlockSparsePrepared, nonclosing_paths_require_a_coarser_output_partition) {
  double beta = 2., Lambda = 40., eps = 1e-12;
  auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false);
  int r = mesh.size();
  nda::vector<double> poles{0.};
  // Two self-adjoint permutations A=(01), B=(12). Their crossed word BABA
  // sends sectors 0->1->2->0, even though each interaction separately closes.
  nda::vector<int> A{1, 0, 2}, B{0, 2, 1};
  std::vector<tensor> scalar_blocks(3, nda::ones<dcomplex>(1, 1, 1));
  BlockOpSymSet a(A, scalar_blocks), b(B, scalar_blocks);
  tensor full_ops = nda::zeros<dcomplex>(2, 3, 3);
  for (int source = 0; source < 3; ++source) {
    full_ops(0, A(source), source) = 1.;
    full_ops(1, B(source), source) = 1.;
  }
  std::vector<tensor> grouped_blocks{full_ops};
  BlockOpSymSet group(nda::vector<int>{0}, grouped_blocks);
  auto coefficients = nda::zeros<dcomplex>(1, 2, 2);
  coefficients(0, 0, 0) = coefficients(0, 1, 1) = 1.;
  BlockOpSymQuartet fine_ops({a, b}, {a, b}, coefficients, nda::vector<long>{0, 1});
  BlockOpSymQuartet coarse_ops({group}, {group}, coefficients, nda::vector<long>{0, 0});
  DiagramEvaluator fine(beta, Lambda, eps, poles, coefficients, fine_ops, 2);
  DiagramEvaluator coarse(beta, Lambda, eps, poles, coefficients, coarse_ops, 2);
  std::vector<tensor> fine_blocks(3, nda::ones<dcomplex>(r, 1, 1));
  BlockDiagOpFun fine_G(fine_blocks, nda::vector<int>{0, 0, 0});
  tensor full_G = nda::zeros<dcomplex>(r, 3, 3);
  for (int t = 0; t < r; ++t)
    for (int i = 0; i < 3; ++i) full_G(t, i, i) = 1.;
  std::vector<tensor> coarse_blocks{full_G};
  BlockDiagOpFun coarse_G(coarse_blocks, nda::vector<int>{0});
  auto assignment = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(
     nda::array<int, 2>{{0, 2}, {1, 3}},
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::forward, -1, -1}, {Direction::forward, 1, 0}}, fine.action.statistics,
                                      poles, triqs_xca::diagram::Kind::self_energy));
  auto sigma = coarse.compute_self_energy(coarse_G, assignment);
  // G=I, both scalar interactions are -1/2, and the ordered integration area
  // is tau^2/2. The raw order-two bosonic sign gives -(tau^2/8)*(BABA+I).
  int t = 0;
  double largest_off_diagonal = 0.;
  for (auto tau : mesh) {
    double factor = -double(tau) * double(tau) / 8.;
    for (int source = 0; source < 3; ++source)
      for (int target = 0; target < 3; ++target) {
        double expected = factor * ((target == source) + (target == (source + 1) % 3));
        EXPECT_LE(std::abs(sigma[0].data()(t, target, source) - expected), 5e-11);
        if (target != source) largest_off_diagonal = std::max(largest_off_diagonal, std::abs(sigma[0].data()(t, target, source)));
      }
    ++t;
  }
  ASSERT_GT(largest_off_diagonal, 0.1);
  try {
    fine.compute_self_energy(fine_G, assignment);
    FAIL() << "an off-diagonal sector contribution must not be accumulated into a diagonal block of the same size";
  } catch (std::invalid_argument const &error) {
    std::string message = error.what();
    EXPECT_NE(message.find("sector"), std::string::npos);
    EXPECT_NE(message.find("partition"), std::string::npos);
  }

  // Recovery after rejection: first-order paths do close for this model.
  auto first_order =
     triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(nda::array<int, 2>{{0, 1}},
                                          triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::forward, -1, -1}},
                                                                           fine.action.statistics, poles, triqs_xca::diagram::Kind::self_energy));
  auto recovered = fine.compute_self_energy(fine_G, first_order);
  for (int sector = 0; sector < 3; ++sector)
    EXPECT_LE(nda::max_element(nda::abs(recovered[sector].data() + 1.)), 1e-12);

  // An unused A interaction leaves candidate nonclosing paths but contributes
  // exactly zero there. It must not cause rejection or pollute the B-only result.
  coefficients(0, 0, 0) = 0.;
  BlockOpSymQuartet b_only_ops({a, b}, {a, b}, coefficients, nda::vector<long>{0, 1});
  DiagramEvaluator b_only(beta, Lambda, eps, poles, coefficients, b_only_ops, 2);
  auto closed = b_only.compute_self_energy(fine_G, assignment);
  t = 0;
  for (auto tau : mesh) {
    for (int sector = 0; sector < 3; ++sector)
      EXPECT_LE(std::abs(closed[sector].data()(t, 0, 0) + double(tau) * double(tau) / 8.), 5e-11);
    ++t;
  }
}

namespace {
  template <typename Backbone>
  concept SelfEnergyInput = requires(DiagramEvaluator &evaluator, BlockDiagOpFun &G, Backbone const &backbone) {
    evaluator.compute_self_energy(G, backbone);
  };

  template <typename Backbone>
  concept CorrelatorInput = requires(DiagramEvaluator &evaluator, BlockDiagOpFun &G, Backbone const &backbone,
                                     std::vector<BlockOp> const &operators) {
    evaluator.eval_correlator(G, backbone, operators, operators, triqs_xca::diagram::CorrelatorStatistics{});
  };
}

TEST(BlockSparsePrepared, evaluator_entry_points_require_the_correct_backbone_kind) {
  using triqs_xca::diagram::SelfEnergyBackbone;
  using triqs_xca::diagram::CorrelatorBackbone;
  static_assert(SelfEnergyInput<SelfEnergyBackbone>);
  static_assert(!SelfEnergyInput<CorrelatorBackbone>);
  static_assert(CorrelatorInput<CorrelatorBackbone>);
  static_assert(!CorrelatorInput<SelfEnergyBackbone>);
}

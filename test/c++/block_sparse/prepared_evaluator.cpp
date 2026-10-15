#include <gtest/gtest.h>
#include <optional>
#include <triqs_xca/block_sparse/dynint.hpp>

#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include "test_utils/evaluators.hpp"

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
  DiagramTopology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  int n = evaluator.n;
  auto const assignment = prepare_assignment(topology, n, AssignmentIndices{{0, 1}, {1}, {n - 1}}, evaluator.hyb.poles, DiagramKind::self_energy);
  long f = (n - 1) + n * (1 + model.hyb_poles.size() * 2);
  auto expected = test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(G, topology.arcs, f), model.ad);
  auto actual = test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(G, assignment), model.ad);
  expect_close(actual, expected);

  auto changed_G = G;
  changed_G *= 1.1;
  auto changed = test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(changed_G, assignment), model.ad);
  expect_close(changed, nda::make_regular((1.1 * 1.1 * 1.1) * expected));
  EXPECT_EQ(assignment.flat_index, -1);
  EXPECT_EQ(assignment.directions(0), 0);
  EXPECT_TRUE(assignment.vertices[0].dagger);
}

TEST(BlockSparsePrepared, logical_correlator_reuses_preparation_with_changed_propagator) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
  BlockDiagOpFun G(test_utils::subspace_blocks(model.G, model.ad, model.mesh));
  DiagramTopology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  int n = evaluator.n;
  auto const assignment = prepare_assignment(topology, n, AssignmentIndices{{1}, {1}, {n - 1}}, evaluator.hyb.poles, DiagramKind::correlator);
  long f = (n - 1) + n * (1 + model.hyb_poles.size());
  CorrelatorBackbone legacy(topology.arcs, n, evaluator.n_int);
  auto op = identity(G);
  auto expected = evaluator.eval_correlator(G, legacy, {op}, {op}, f, false);
  auto actual = evaluator.eval_correlator(G, assignment, {op}, {op}, false);
  expect_close(actual, expected);

  auto changed_G = G;
  changed_G *= 1.1;
  auto changed = evaluator.eval_correlator(changed_G, assignment, {op}, {op}, false);
  expect_close(changed, nda::make_regular((1.1 * 1.1 * 1.1 * 1.1) * expected));
  EXPECT_EQ(assignment.flat_index, -1);
  EXPECT_EQ(assignment.directions(0), -1);
  EXPECT_EQ(assignment.vertices[0].orbital, -1);
}


TEST(BlockSparsePrepared, bound_operands_borrow_model_data_and_own_path_metadata) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  auto [operators, labels] = block_sparse::dynint::get_operators_and_interactions(model.ad, model.hyb_coeffs, model.dynint_coeffs, model.dynint_ops);
  auto coefficients = hyb::get_extended_coefficients(model.hyb_coeffs, model.dynint_coeffs);
  hyb::Hybridization hybridization(model.mesh, model.hyb_poles, coefficients);
  BlockDiagOpFun G(test_utils::subspace_blocks(model.G, model.ad, model.mesh));
  auto changed_G = G;
  changed_G *= 1.3;
  int n = coefficients.extent(1);
  auto statistics = fermion_boson_statistics(model.n_hyb(), model.n_int());
  DiagramTopology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  constexpr auto all = nda::range::all;
  for (int direction : {0, 1}) {
    auto assignment = prepare_assignment(topology, n, AssignmentIndices{{direction, 1}, {1}, {n - 1}}, hybridization.poles, DiagramKind::self_energy);
    auto signs = prepare_endpoint_signs(assignment, statistics);
    SelfEnergyPath workspace(2);
    std::optional<SelfEnergyContribution> retained;
    nda::vector<int> dimensions;
    int count = 0;
    visit_self_energy_paths(G, assignment, operators, workspace, [&](auto const &path) {
      auto bound = bind_self_energy_contribution(G, assignment, path, operators, hybridization, statistics, signs);
      auto changed = bind_self_energy_contribution(changed_G, assignment, path, operators, hybridization, statistics, signs);
      ++count;
      ASSERT_EQ(bound.vertices.size(), 4);
      ASSERT_EQ(bound.propagators.size(), 3);
      for (int e = 0; e < 3; ++e) {
        EXPECT_EQ(bound.propagators[e].data(), G.get_block(path.propagator_blocks(e)).data());
        expect_close(tensor(changed.propagators[e]), nda::make_regular(1.3 * bound.propagators[e]));
      }
      for (int v : {1, 3}) {
        EXPECT_EQ(bound.vertices[v].extent(0), path.dimensions(v + 1));
        EXPECT_EQ(bound.vertices[v].extent(1), path.dimensions(v));
      }
      EXPECT_EQ(bound.endpoints.coefficients.size(), bound.endpoints.late.size());
      for (int mu = 0; mu < bound.endpoints.late.size(); ++mu) {
        auto const &late = bound.endpoints.late[mu];
        EXPECT_EQ(late.statistics, statistics[late.coefficient_index]);
        for (auto const &early : bound.endpoints.early) {
          EXPECT_EQ(early.statistics, statistics[early.coefficient_index]);
          auto expected = direction == 1 ? hybridization.values(all, late.coefficient_index, early.coefficient_index)
                                         : hybridization.values_reflect(all, early.coefficient_index, late.coefficient_index);
          auto actual = bound.endpoints.coefficients[mu](all, early.coefficient_index);
          EXPECT_EQ(nda::max_element(nda::abs(actual - expected)), 0.0);
        }
      }
      if (!retained) {
        dimensions = path.dimensions;
        retained.emplace(bound);
      }
    });
    ASSERT_GT(count, 0);
    workspace.dimensions = -99;
    ASSERT_TRUE(retained);
    EXPECT_EQ(retained->path.dimensions, dimensions);
    EXPECT_EQ(retained->endpoints.reflection_sign, direction == 1 ? 1 : -1);
  }
}

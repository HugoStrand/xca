#include "triqs_xca/diagram/correlator.hpp"
#include <gtest/gtest.h>

#include <triqs_xca/dense/diagram_evaluator.hpp>
#include "test_utils/evaluators.hpp"

namespace {
using namespace triqs_xca;
using nda::dcomplex;
using tensor = nda::array<dcomplex, 3>;

void expect_close(tensor const &actual, tensor const &expected) {
  double scale = nda::max_element(nda::abs(expected));
  ASSERT_GT(scale, 1e-14) << "the selected contribution must be resolved";
  EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 1e-12 * scale);
}
}

TEST(DensePrepared, logical_self_energy_and_pair_reuse_the_same_assignment) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  dense::DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  auto G = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>{std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(model.mesh, model.G)}};
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  int n = evaluator.n;
  auto const assignment = triqs_xca::diagram::prepare_diagram(topology, n, triqs_xca::diagram::Indices{{0, 1}, {1}, {n - 1}}, model.hyb_poles, triqs_xca::diagram::Kind::self_energy);
  long f = (n - 1) + n * (1 + model.hyb_poles.size() * 2);
  long partner = f + n * model.hyb_poles.size();
  auto expected = tensor(evaluator.compute_self_energy(G, topology.arcs, f)[0].data());
  auto expected_pair = nda::make_regular(expected + evaluator.compute_self_energy(G, topology.arcs, partner)[0].data());

  evaluator.reset();
  evaluator.eval_self_energy(model.G, assignment);
  expect_close(evaluator.Sigma, expected);
  evaluator.reset();
  evaluator.eval_self_energy_pair(model.G, assignment);
  expect_close(evaluator.Sigma, expected_pair);
  EXPECT_EQ(assignment.directions(0), 0);
  EXPECT_TRUE(assignment.vertices[0].adjoint);

  // There are three propagator edges at order two. Reusing preparation with a
  // different G must not reuse old numerical data or an old accumulator.
  evaluator.reset();
  auto changed_G = nda::make_regular(1.1 * model.G);
  evaluator.eval_self_energy(changed_G, assignment);
  expect_close(evaluator.Sigma, nda::make_regular((1.1 * 1.1 * 1.1) * expected));
}

TEST(DensePrepared, logical_correlator_has_no_flat_index_or_mutable_backbone) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  dense::DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  auto G = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>{std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(model.mesh, model.G)}};
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  int n = evaluator.n;
  auto const assignment = triqs_xca::diagram::prepare_diagram(topology, n, triqs_xca::diagram::Indices{{1}, {1}, {n - 1}}, model.hyb_poles, triqs_xca::diagram::Kind::correlator);
  long f = (n - 1) + n * (1 + model.hyb_poles.size());
  auto expected = evaluator.compute_single_ptcle_gf(G, topology.arcs, f);
  auto annihilators = nda::make_regular(evaluator.Fset.Fs(nda::range(evaluator.n_hyb), nda::range::all, nda::range::all));
  auto creators = nda::make_regular(evaluator.Fset.F_dags(nda::range(evaluator.n_hyb), nda::range::all, nda::range::all));
  auto actual = evaluator.eval_correlator(model.G, assignment, annihilators, creators, triqs_xca::diagram::CorrelatorStatistics{});
  expect_close(actual, expected);
  auto changed_G = nda::make_regular(1.1 * model.G);
  auto changed = evaluator.eval_correlator(changed_G, assignment, annihilators, creators, triqs_xca::diagram::CorrelatorStatistics{});
  expect_close(changed, nda::make_regular((1.1 * 1.1 * 1.1 * 1.1) * expected));
  EXPECT_EQ(assignment.directions(0), -1);
  EXPECT_EQ(assignment.vertices[0].flavour, -1);
}

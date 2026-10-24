#include <gtest/gtest.h>
#include <triqs_xca/dense/action.hpp>
#include <triqs_xca/block_sparse/action.hpp>
#include <triqs_xca/arc_data.hpp>
#include "test_utils/evaluators.hpp"

using namespace triqs_xca;

TEST(Action, atomic_and_numerical_inputs_share_scalar_data_and_rebuild_bars) {
  auto model = test_utils::build_eval_model("unequal_sets");
  auto dense = dense::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad_flat);
  auto sparse = block_sparse::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad);
  ArcData reference(model.mesh, model.hyb_poles, model.hyb_coeffs);
  auto check = [&](auto const &action) {
    EXPECT_EQ(action.poles, model.hyb_poles);
    EXPECT_EQ(action.pole_coeffs, model.hyb_coeffs);
    EXPECT_EQ(action.tau_values, reference.values);
    EXPECT_EQ(action.tau_values_reflect, reference.values_reflect);
    EXPECT_EQ(action.kernel.at_tau_plus, reference.kernel_values.at_tau_plus);
    EXPECT_EQ(action.kernel.at_tau_minus, reference.kernel_values.at_tau_minus);
    EXPECT_EQ(action.n_hyb, model.n_hyb());
  };
  check(dense);
  check(sparse);

  // Deliberately stale bars: construction must use the new Action coefficients.
  auto coefficients = nda::make_regular(2.0 * model.hyb_coeffs);
  auto new_dense = dense::make_action(model.mesh, model.hyb_poles, coefficients, dense.operators);
  auto new_sparse = block_sparse::make_action(model.mesh, model.hyb_poles, coefficients, sparse.operators);
  EXPECT_LT(nda::max_element(nda::abs(new_dense.operators.F_dag_bars - 2.0 * dense.operators.F_dag_bars)), 1e-14);
  for (int g = 0; g < sparse.operators.F_dag_bars.size(); ++g)
    for (int b = 0; b < sparse.block_dims.size(); ++b)
      if (sparse.operators.F_dag_bars[g].get_block_index(b) >= 0)
        EXPECT_LT(nda::max_element(nda::abs(new_sparse.operators.F_dag_bars[g].get_block(b)
                       - 2.0 * sparse.operators.F_dag_bars[g].get_block(b))), 1e-14);
  auto moved = std::move(new_dense);
  EXPECT_EQ(moved.pole_coeffs, coefficients);
}

TEST(Action, validates_metadata_before_scalar_evaluation) {
  auto model = test_utils::build_eval_model("unequal_sets");
  auto operators = dense::atom_diag::get_operators(model.ad_flat, model.hyb_coeffs);
  auto wrong_poles = nda::vector<double>{1.};
  EXPECT_THROW(dense::make_action(model.mesh, wrong_poles, model.hyb_coeffs, operators), std::invalid_argument);
  EXPECT_THROW(dense::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, operators,
                                 std::vector<diagram::Statistics>{}, 0), std::invalid_argument);
}

TEST(Action, mixed_and_empty_dynamical_inputs_keep_flavour_order) {
  for (auto const &name : {"unequal_sets", "unequal_sets_nA0"}) {
    auto model = test_utils::build_eval_model(name);
    auto dense = dense::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
    auto sparse = block_sparse::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad, model.dynint_ops, model.dynint_coeffs);
    auto expected = hyb::get_extended_coefficients(model.hyb_coeffs, model.dynint_coeffs);
    EXPECT_EQ(dense.pole_coeffs, expected);
    EXPECT_EQ(sparse.pole_coeffs, expected);
    EXPECT_EQ(dense.statistics, sparse.statistics);
    EXPECT_EQ(dense.statistics.size(), model.n_hyb() + model.n_int());
    for (int i = 0; i < dense.statistics.size(); ++i)
      EXPECT_EQ(dense.statistics[i], i < model.n_hyb() ? diagram::Statistics::fermionic : diagram::Statistics::bosonic);
    auto old = dense::dynint::get_operators_and_interactions(model.ad_flat, model.hyb_coeffs, model.dynint_coeffs, model.dynint_ops);
    EXPECT_EQ(dense.operators.F_dag_bars, old.F_dag_bars);
    EXPECT_EQ(dense.operators.F_bars_refl, old.F_bars_refl);
  }
}

TEST(Action, explicit_dimensions_keep_sectors_where_all_operators_vanish) {
  using nda::dcomplex;
  using tensor = nda::array<dcomplex, 3>;
  using gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>;
  using bgf = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-12, false);
  auto coefficients = nda::ones<dcomplex>(1, 1, 1);
  std::vector<tensor> blocks{nda::ones<dcomplex>(1, 1, 1), nda::zeros<dcomplex>(1, 3, 3)};
  block_sparse::BlockOpSymSet op(nda::vector<int>{0, -1}, blocks);
  block_sparse::BlockOpSymQuartet quartet({op}, {op}, coefficients, nda::vector<long>{0});
  auto model = block_sparse::make_action(mesh, nda::vector<double>{0.}, coefficients, quartet,
                                        {diagram::Statistics::bosonic}, 0, nda::vector<long>{1, 3});
  block_sparse::DiagramEvaluator evaluator(std::move(model));
  EXPECT_EQ(evaluator.Nmax, 3);
  EXPECT_EQ(evaluator.n_hyb, 0);
  bgf G{std::vector{gf(mesh, nda::ones<dcomplex>(mesh.size(), 1, 1)), gf(mesh, nda::ones<dcomplex>(mesh.size(), 3, 3))}};
  auto result = evaluator.compute_self_energy(G, nda::array<int, 2>{{0, 1}});
  EXPECT_EQ(nda::max_element(nda::abs(result[1].data())), 0.);
  bgf bad{std::vector{G[0], gf(mesh, nda::ones<dcomplex>(mesh.size(), 2, 2))}};
  EXPECT_THROW(evaluator.compute_self_energy(bad, nda::array<int, 2>{{0, 1}}), std::invalid_argument);
}

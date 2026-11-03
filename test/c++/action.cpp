#include "test_utils/prepared_evaluators.hpp"
#include <gtest/gtest.h>
#include <triqs_xca/dense/action.hpp>
#include <triqs_xca/block_sparse/action.hpp>
#include "test_utils/evaluators.hpp"

using namespace triqs_xca;

TEST(Action, atomic_and_numerical_inputs_share_scalar_data_and_rebuild_bars) {
  auto model = test_utils::build_eval_model("unequal_sets");
  auto dense = dense::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad_flat);
  auto sparse = block_sparse::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad);
  auto values = coefs2vals(model.mesh.beta(), model.mesh.dlr_it(), model.hyb_coeffs, model.hyb_poles);
  auto reflected = model.mesh.dlr_it().reflect(values);
  auto check = [&](auto const &action) {
    EXPECT_EQ(action.poles, model.hyb_poles);
    EXPECT_EQ(action.pole_coeffs, model.hyb_coeffs);
    EXPECT_EQ(action.tau_values, values);
    EXPECT_EQ(action.tau_values_reflect, reflected);
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
                                 std::vector<Statistics>{}, 0), std::invalid_argument);
}

TEST(Action, mixed_and_empty_dynamical_inputs_keep_flavour_order) {
  for (auto const &name : {"unequal_sets", "unequal_sets_nA0"}) {
    auto model = test_utils::build_eval_model(name);
    auto dense = dense::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
    auto sparse = block_sparse::make_action(model.mesh, model.hyb_poles, model.hyb_coeffs, model.ad, model.dynint_ops, model.dynint_coeffs);
    auto expected = get_extended_coefficients(model.hyb_coeffs, model.dynint_coeffs);
    EXPECT_EQ(dense.pole_coeffs, expected);
    EXPECT_EQ(sparse.pole_coeffs, expected);
    EXPECT_EQ(dense.statistics, sparse.statistics);
    EXPECT_EQ(dense.statistics.size(), model.n_hyb() + model.n_int());
    for (int i = 0; i < dense.statistics.size(); ++i)
      EXPECT_EQ(dense.statistics[i], i < model.n_hyb() ? Statistics::fermionic : Statistics::bosonic);
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
                                        {Statistics::bosonic}, 0, nda::vector<long>{1, 3});
  block_sparse::DiagramEvaluator evaluator(std::move(model));
  EXPECT_EQ(nda::max_element(evaluator.action.block_dims), 3);
  EXPECT_EQ(evaluator.action.n_hyb, 0);
  bgf G{std::vector{gf(mesh, nda::ones<dcomplex>(mesh.size(), 1, 1)), gf(mesh, nda::ones<dcomplex>(mesh.size(), 3, 3))}};
  auto result = evaluator.compute_self_energy(G, nda::array<int, 2>{{0, 1}});
  EXPECT_EQ(nda::max_element(nda::abs(result[1].data())), 0.);
  bgf bad{std::vector{G[0], gf(mesh, nda::ones<dcomplex>(mesh.size(), 2, 2))}};
  EXPECT_THROW(evaluator.compute_self_energy(bad, nda::array<int, 2>{{0, 1}}), std::invalid_argument);
}

TEST(Action, dense_legacy_constructor_rejects_a_different_DLR_grid) {
  auto model = test_utils::build_eval_model("unequal_sets");
  auto operators = dense::atom_diag::get_operators(model.ad_flat, model.hyb_coeffs);
  auto const &ops = model.mesh.dlr_it();
  EXPECT_NO_THROW(dense::DiagramEvaluator(model.mesh.beta(), model.mesh.eps(), ops, model.hyb_poles, model.hyb_coeffs, operators));
  cppdlr::imtime_ops custom(ops.lambda(), nda::vector<double>{0.});
  EXPECT_THROW(dense::DiagramEvaluator(model.mesh.beta(), model.mesh.eps(), custom, model.hyb_poles, model.hyb_coeffs, operators),
               std::invalid_argument);
  auto mesh = triqs::mesh::dlr_imtime(model.mesh.beta(), triqs::mesh::Fermion, 20., 1e-10, true);
  dense::DiagramEvaluator evaluator(dense::make_action(mesh, model.hyb_poles, model.hyb_coeffs, operators));
  EXPECT_EQ(evaluator.action.tau_mesh, mesh);
  EXPECT_EQ(evaluator.action.tau_mesh.dlr_it().get_itnodes(), mesh.dlr_it().get_itnodes());
}

TEST(Action, evaluator_copies_own_operator_data_and_outlive_their_source) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  auto dense = [&] {
    dense::DiagramEvaluator source(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
    auto copy = source;
    EXPECT_NE(copy.action.operators.Fs.data(), source.action.operators.Fs.data());
    EXPECT_NE(copy.action.pole_coeffs.data(), source.action.pole_coeffs.data());
    return copy;
  }();
  auto sparse = [&] {
    block_sparse::DiagramEvaluator source(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
    auto copy = source;
    EXPECT_NE(copy.action.pole_coeffs.data(), source.action.pole_coeffs.data());
    return copy;
  }();
  auto G_dense = test_utils::subspace_blocks(model.G, model.ad_flat, model.mesh);
  auto G_sparse = test_utils::subspace_blocks(model.G, model.ad, model.mesh);
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  auto sigma_dense = triqs_xca::test_utils::fixture_self_energy(dense, G_dense, topology, itertools::range(0L, 8L));
  auto sigma_sparse = triqs_xca::test_utils::fixture_self_energy(sparse, G_sparse, topology, itertools::range(0L, 8L));
  auto expected = test_utils::get_tensor_in_full_hilbert_space(sigma_sparse, model.ad);
  EXPECT_LT(nda::max_element(nda::abs(sigma_dense[0].data() - expected)), 1e-12);
}

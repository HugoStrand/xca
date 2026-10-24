#include <cmath>
#include <gtest/gtest.h>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include "test_utils/evaluators.hpp"

namespace {
  using namespace triqs_xca;
  using nda::dcomplex;
  using tensor = nda::array<dcomplex, 3>;
  using gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>;
  using bgf = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  constexpr auto all = nda::range::all;

  void expect_close(tensor const &actual, tensor const &expected) {
    double scale = nda::max_element(nda::abs(expected));
    ASSERT_GT(scale, 1e-4) << "the comparison must resolve a nonzero contribution";
    EXPECT_LT(nda::max_element(nda::abs(actual - expected)), 1e-10 * scale);
  }
}

TEST(EvaluatorConventions, time_reflection_and_each_endpoint_direction) {
  double beta = 2., Lambda = 40., eps = 1e-12;
  auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false);
  auto itops = mesh.dlr_it();
  auto tau = cppdlr::rel2abs(itops.get_itnodes());
  nda::vector<double> physical_poles{-0.8, 1.3};
  tensor coefficients(2, 2, 2);
  coefficients(0, all, all) = nda::array<dcomplex, 2>{{dcomplex(2, 1), dcomplex(3, -2)}, {dcomplex(-1, 4), dcomplex(5, 3)}};
  coefficients(1, all, all) = nda::array<dcomplex, 2>{{dcomplex(-1, 2), dcomplex(2, 3)}, {dcomplex(4, -1), dcomplex(1, -2)}};
  // Two fermions in |00>, |10>, |01>, |11>; c1|11> = -|10>.
  tensor F = nda::zeros<dcomplex>(2, 4, 4), Fdag = F;
  F(0, 0, 1) = F(0, 2, 3) = F(1, 0, 2) = 1.;
  F(1, 1, 3) = -1.;
  for (int a = 0; a < 2; ++a) Fdag(a, all, all) = nda::dagger(F(a, all, all));
  dense::FSet fset(F, Fdag, coefficients);
  dense::DiagramEvaluator dense(beta, eps, itops, physical_poles, coefficients, fset);
  nda::vector<int> connection{0};
  std::vector<tensor> blocks{F}, dag_blocks{Fdag};
  block_sparse::BlockOpSymSet grouped_F(connection, blocks), grouped_Fdag(connection, dag_blocks);
  block_sparse::BlockOpSymQuartet quartet({grouped_F}, {grouped_Fdag}, coefficients, nda::vector<long>{0, 0});
  block_sparse::DiagramEvaluator sparse(beta, Lambda, eps, physical_poles, coefficients, quartet);

  tensor values = nda::zeros<dcomplex>(mesh.size(), 2, 2), reflected = values;
  for (int t = 0; t < mesh.size(); ++t)
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b)
        for (int p = 0; p < physical_poles.size(); ++p) {
          double omega = beta * physical_poles(p);
          values(t, a, b) -= coefficients(p, a, b) * std::exp(-tau(t) * omega) / (1. + std::exp(-omega));
          reflected(t, a, b) -= coefficients(p, a, b) * std::exp(-(1. - tau(t)) * omega) / (1. + std::exp(-omega));
        }
  for (auto const *data : {&dense.arc_data, &sparse.arc_data}) {
    expect_close(data->values, values);
    expect_close(data->values_reflect, reflected);
  }

  tensor G = nda::zeros<dcomplex>(mesh.size(), 4, 4);
  for (int t = 0; t < mesh.size(); ++t)
    for (int i = 0; i < 4; ++i) G(t, i, i) = dcomplex(1. + i + 0.2 * tau(t), 0.1 * i);
  bgf propagator{std::vector{gf(mesh, G)}};
  nda::array<int, 2> topology{{0, 1}};
  tensor sum = nda::zeros<dcomplex>(mesh.size(), 4, 4);
  for (long direction : {0, 1}) {
    SCOPED_TRACE(direction);
    tensor expected = nda::zeros<dcomplex>(mesh.size(), 4, 4);
    // First order: forward = sum C_mu,kap F_mu^dag G F_kap.
    // Backward = sum C_kap,mu(beta-tau) F_mu G F_kap^dag:
    // the orientation minus and external-direction minus cancel.
    for (int t = 0; t < mesh.size(); ++t)
      for (int mu = 0; mu < 2; ++mu)
        for (int kap = 0; kap < 2; ++kap) {
          auto const &left = direction == 1 ? Fdag : F;
          auto const &right = direction == 1 ? F : Fdag;
          dcomplex coefficient = direction == 1 ? values(t, mu, kap) : reflected(t, kap, mu);
          expected(t, all, all) += coefficient * nda::linalg::matmul(left(mu, all, all), nda::linalg::matmul(G(t, all, all), right(kap, all, all)));
        }
    expect_close(tensor(dense.compute_self_energy(propagator, topology, direction)[0].data()), expected);
    expect_close(tensor(sparse.compute_self_energy(propagator, topology, direction)[0].data()), expected);
    sum += expected;
  }
  expect_close(tensor(dense.compute_self_energy_by_pairs(propagator, topology)[0].data()), sum);
}

TEST(EvaluatorConventions, all_constructor_routes_accept_physical_poles) {
  auto model = test_utils::build_eval_model("unequal_sets");
  double Lambda = 40., eps = 1e-12;
  for (double beta : {1.5, 3.}) {
    SCOPED_TRACE(beta);
    auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false);
    auto itops = mesh.dlr_it();
    // Use the same single-sector realization so matrix layouts coincide.
    dense::DiagramEvaluator dense_atom(model.hyb_poles, model.hyb_coeffs, mesh, model.ad_flat);
    block_sparse::DiagramEvaluator sparse_atom(model.hyb_poles, model.hyb_coeffs, mesh, model.ad_flat);
    dense::DiagramEvaluator dense_ops(beta, eps, itops, model.hyb_poles, model.hyb_coeffs, dense_atom.Fset);
    auto quartet = std::get<0>(block_sparse::atom_diag::get_operators(model.ad_flat, model.hyb_coeffs));
    block_sparse::DiagramEvaluator sparse_ops(beta, Lambda, eps, model.hyb_poles, model.hyb_coeffs, quartet);

    for (auto const *data : {&dense_atom.arc_data, &sparse_atom.arc_data, &dense_ops.arc_data, &sparse_ops.arc_data}) {
      for (int p = 0; p < model.hyb_poles.size(); ++p) EXPECT_DOUBLE_EQ(data->poles(p), beta * model.hyb_poles(p));
      EXPECT_EQ(nda::max_element(nda::abs(data->coeffs - model.hyb_coeffs)), 0.);
      expect_close(data->values, dense_atom.arc_data.values);
      expect_close(data->values_reflect, dense_atom.arc_data.values_reflect);
      auto const &actual = data->kernel_values;
      auto const &expected = dense_atom.arc_data.kernel_values;
      EXPECT_LT(nda::max_element(nda::abs(actual.at_tau_plus - expected.at_tau_plus)), 1e-13);
      EXPECT_LT(nda::max_element(nda::abs(actual.at_tau_minus - expected.at_tau_minus)), 1e-13);
      EXPECT_LT(nda::max_element(nda::abs(actual.at_zero_plus - expected.at_zero_plus)), 1e-13);
      EXPECT_LT(nda::max_element(nda::abs(actual.at_zero_minus - expected.at_zero_minus)), 1e-13);
    }

    int N = model.ad_flat.get_full_hilbert_space_dim();
    tensor G = nda::zeros<dcomplex>(mesh.size(), N, N);
    for (int t = 0; t < mesh.size(); ++t)
      for (int i = 0; i < N; ++i) G(t, i, i) = 1. + 0.3 * i;
    bgf propagator{std::vector{gf(mesh, G)}};
    // First order resolves the time-valued coefficients; second order also
    // resolves barred operators, kernel placement and convolution factors.
    for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}}) {
      std::vector<long> selected{0, 1};
      if (topology.extent(0) == 2) {
        selected.clear();
        for (int direction = 0; direction < 4; ++direction)
          for (int pole = 0; pole < model.hyb_poles.size(); ++pole)
            selected.push_back(model.n_hyb() * (pole + model.hyb_poles.size() * direction));
      }
      for (long f : selected) {
        SCOPED_TRACE(::testing::Message() << "order=" << topology.extent(0) << ", index=" << f);
        tensor expected = dense_atom.compute_self_energy(propagator, topology, f)[0].data();
        expect_close(tensor(dense_ops.compute_self_energy(propagator, topology, f)[0].data()), expected);
        expect_close(tensor(sparse_atom.compute_self_energy(propagator, topology, f)[0].data()), expected);
        expect_close(tensor(sparse_ops.compute_self_energy(propagator, topology, f)[0].data()), expected);
      }
    }
  }
}

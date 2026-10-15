#include <gtest/gtest.h>
#include <cmath>
#include <limits>

#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>

namespace {
using namespace triqs_xca;
using namespace triqs_xca::block_sparse;
using nda::dcomplex;
using tensor = nda::array<dcomplex, 3>;
using gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>;
using bgf = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;

void expect_scaled(tensor const &actual, tensor const &expected) {
  double scale = nda::max_element(nda::abs(expected));
  if (scale == 0) EXPECT_EQ(nda::max_element(nda::abs(actual)), 0.);
  else {
    EXPECT_GT(nda::max_element(nda::abs(actual)), 0.);
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 2e-11 * scale);
  }
}
}

TEST(NumericalZero, imported_and_refreshed_masks_preserve_every_nonzero_value) {
  auto mesh = triqs::mesh::dlr_imtime(2., triqs::mesh::Fermion, 20., 1e-12, false);
  for (double magnitude : {0., 1e-18, std::numeric_limits<double>::denorm_min()}) {
    tensor values = nda::zeros<dcomplex>(mesh.size(), 1, 1);
    values(0, 0, 0) = dcomplex(0., magnitude);
    BlockDiagOpFun imported(bgf{std::vector{gf(mesh, values)}});
    EXPECT_EQ(imported.get_zero_block_index(0), magnitude == 0 ? -1 : 0);
    imported.set_zero_block_indices();
    EXPECT_EQ(imported.get_zero_block_index(0), magnitude == 0 ? -1 : 0);
    EXPECT_EQ(imported.get_block(0)(0, 0, 0), values(0, 0, 0));
  }
}

TEST(NumericalZero, cancellation_residual_survives_mask_refresh_and_later_accumulation) {
  double large = std::ldexp(1., -40), residual = std::ldexp(1., -60);
  BlockDiagOpFun sum(1, nda::vector<int>{1});
  tensor term(1, 1, 1);
  term = large;
  sum.add_block(0, term);
  term = -(large - residual); // exactly representable binary cancellation
  sum.add_block(0, term);
  ASSERT_EQ(sum.get_block(0)(0, 0, 0), dcomplex(residual));
  sum.set_zero_block_indices();
  EXPECT_EQ(sum.get_zero_block_index(0), 0);
  auto by_addition = sum;
  term = residual / 2;
  sum.add_block(0, term);
  BlockDiagOpFun other(1, nda::vector<int>{1});
  other.add_block(0, term);
  by_addition += other;
  EXPECT_EQ(sum.get_block(0)(0, 0, 0), dcomplex(1.5 * residual));
  EXPECT_EQ(by_addition.get_block(0)(0, 0, 0), dcomplex(1.5 * residual));
  term = -1.5 * residual;
  sum.add_block(0, term);
  sum.set_zero_block_indices();
  EXPECT_EQ(sum.get_zero_block_index(0), -1); // an exact zero is still removable
}

TEST(NumericalZero, one_fermion_closed_forms_hold_below_the_old_cutoff) {
  double beta = 2., Lambda = 40., eps = 1e-12;
  auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false);
  auto itops = mesh.dlr_it();
  int r = mesh.size();
  nda::array<int, 2> topology{{0, 1}};
  nda::vector<double> poles{0.};
  // Explicit empty/occupied basis: c maps sector 1 to 0, c^dag maps 0 to 1.
  nda::vector<int> down{-1, 0}, up{1, -1};
  std::vector<tensor> annihilation{nda::zeros<dcomplex>(1, 1, 1), nda::ones<dcomplex>(1, 1, 1)};
  std::vector<tensor> creation{nda::ones<dcomplex>(1, 1, 1), nda::zeros<dcomplex>(1, 1, 1)};
  BlockOpSymSet F(down, annihilation), Fdag(up, creation);
  tensor dense_F = nda::zeros<dcomplex>(1, 2, 2), dense_Fdag = nda::zeros<dcomplex>(1, 2, 2);
  dense_F(0, 0, 1) = dense_Fdag(0, 1, 0) = 1.;

  for (auto [g_scale, c_scale] : {std::pair{1., 1.}, {1e-18, 1.}, {1., 1e-18}, {1e-10, 1e-10}, {0., 1.}, {1., 0.}}) {
    SCOPED_TRACE(::testing::Message() << "G scale=" << g_scale << ", coefficient scale=" << c_scale);
    tensor coefficients(1, 1, 1);
    coefficients = c_scale;
    BlockOpSymQuartet quartet({F}, {Fdag}, coefficients, nda::vector<long>{0});
    block_sparse::DiagramEvaluator sparse(beta, Lambda, eps, poles, coefficients, quartet);
    dense::FSet fset(dense_F, dense_Fdag, coefficients);
    dense::DiagramEvaluator dense(beta, eps, itops, poles, coefficients, fset);
    dcomplex g0 = g_scale * dcomplex(1.25, 0.5), g1 = g_scale * dcomplex(-0.75, 0.2);
    tensor G0(r, 1, 1), G1(r, 1, 1), full = nda::zeros<dcomplex>(r, 2, 2);
    G0 = g0;
    G1 = g1;
    for (int t = 0; t < r; ++t) { full(t, 0, 0) = g0; full(t, 1, 1) = g1; }
    bgf G_blocks{std::vector{gf(mesh, G0), gf(mesh, G1)}};
    bgf G_dense{std::vector{gf(mesh, full)}};

    // Delta(tau)=C*K(tau,0)=-C/2. Direct first-order matrix multiplication
    // gives Sigma_empty=Delta*G_occupied and Sigma_occupied=Delta*G_empty.
    tensor expected0(r, 1, 1), expected1(r, 1, 1), expected_full = nda::zeros<dcomplex>(r, 2, 2);
    expected0 = -0.5 * c_scale * g1;
    expected1 = -0.5 * c_scale * g0;
    for (int t = 0; t < r; ++t) {
      expected_full(t, 0, 0) = expected0(t, 0, 0);
      expected_full(t, 1, 1) = expected1(t, 0, 0);
    }
    auto sigma = sparse.compute_self_energy(G_blocks, topology);
    expect_scaled(tensor(sigma[0].data()), expected0);
    expect_scaled(tensor(sigma[1].data()), expected1);
    expect_scaled(tensor(dense.compute_self_energy(G_dense, topology)[0].data()), expected_full);
    // Tr[G(beta-tau) c G(tau) c^dag]=G_occupied*G_empty for these constant Gs.
    tensor expected_spgf(r, 1, 1);
    expected_spgf = g1 * g0;
    expect_scaled(sparse.compute_single_ptcle_gf(G_blocks, topology), expected_spgf);
    expect_scaled(dense.compute_single_ptcle_gf(G_dense, topology), expected_spgf);
  }
}

#include <string>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>

#include "test_utils/block_sparse.hpp"

using nda::dcomplex;

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;

using triqs_xca::block_sparse::atom_diag::get_operators;
using triqs_xca::block_sparse::DiagramEvaluator;
using triqs_xca::hyb::get_extended_coefficients;
using triqs_xca::block_sparse::dynint::get_operators_and_interactions;
namespace test_utils = triqs_xca::test_utils;

/**
 * @file block_sparse/dynint_spin_flip.cpp
 *
 * @brief The block-sparse symmetry sets of the non-block-diagonal interaction operators S+ and S-
 *
 * @details Density operators are block diagonal and hermitian, while S+ and S- map (N_up, N_do) -> (N_up +- 1, N_do -+ 1), so on the
 * block-sparse side they exercise the connection-row and inverse-map machinery of resolve_dynint_op(), with F_dags(n_hyb + i) != Fs(n_hyb + i).
 * The self-energy and the single-particle Green's function with S+ and S- are compared across evaluators in sigma_spgf_all_evals.cpp, and
 * the physical convention of dynint_coeffs is checked against ED in test/python/dynint_transverse.py.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  constexpr int p_poles = 2;

  struct SpinFlipModel {
    triqs::atom_diag::atom_diag<true> ad; // sym_ops = {N_up, N_do}
    nda::vector<double> hyb_poles;
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> dynint_coeffs;
    std::vector<many_body_operator_complex> dynint_ops;
  };

  many_body_operator_complex S_minus(int norb) {
    many_body_operator_complex op;
    for (int i = 0; i < norb; ++i) op += c_dag("do", i) * c("up", i);
    return op;
  }

  many_body_operator_complex S_plus(int norb) {
    many_body_operator_complex op;
    for (int i = 0; i < norb; ++i) op += c_dag("up", i) * c("do", i);
    return op;
  }

  /**
   * @brief sz_resolved_atom_diag_helper with dynint_ops = [S-, S+]
   *
   * @details On this fixture S+ and S- are single-target operators, since N_up and N_do are separately conserved, and dynint_coeffs is
   * diagonal to encode S+ D S- and S- D S+ separately.
   */
  SpinFlipModel spin_flip_model(int norb) {
    auto ad = test_utils::sz_resolved_atom_diag_helper(norb, true);

    int n_hyb     = static_cast<int>(ad.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
    auto hyb      = test_utils::sym_set_diagonal_hyb(labels_f, p_poles);

    std::vector<many_body_operator_complex> ops{S_minus(norb), S_plus(norb)};

    auto d = nda::zeros<dcomplex>(p_poles, 2, 2);
    for (int l = 0; l < p_poles; ++l) {
      d(l, 0, 0) = 0.61 - 0.07 * l;
      d(l, 1, 1) = 0.43 + 0.11 * l;
    }

    nda::vector<double> hyb_poles = {1.3, -0.8};
    return {ad, hyb_poles, hyb, d, ops};
  }

} // namespace

/**
 * @brief Check compatible grouping and exact member support for S+ and S-, and construct an evaluator
 */
TEST(BlockSparseDynintSpinFlip, spin_flip_ops_construct_a_valid_evaluator) {

  for (int norb : {1, 2}) {
    SCOPED_TRACE("norb = " + std::to_string(norb));
    auto m    = spin_flip_model(norb);
    int n_hyb = static_cast<int>(m.ad.get_fops().size());
    ASSERT_EQ(n_hyb, 2 * norb);

    auto [Fq, labels] = get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops);
    auto ext          = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);

    // With one orbital their supports are disjoint and their union is a swap.
    // With two orbitals the maps disagree on shared sources and must stay separate.
    ASSERT_EQ(labels.size(), n_hyb + 2);
    if (norb == 1) EXPECT_EQ(labels(n_hyb), labels(n_hyb + 1));
    else EXPECT_NE(labels(n_hyb), labels(n_hyb + 1));
    triqs_xca::block_sparse::OperatorSupportTable support(Fq);
    for (int a = 0; a < 2; ++a) {
      auto expected = m.ad.get_op_mat(m.dynint_ops[a]).connection;
      for (int b = 0; b < m.ad.n_subspaces(); ++b)
        EXPECT_EQ(support.plain[n_hyb + a].single_target(b), expected(b));
    }
    for (int i = 0; i < n_hyb; ++i) {
      EXPECT_NE(labels(i), labels(n_hyb)) << "fermionic flavour " << i << " merged with the S- set";
      EXPECT_NE(labels(i), labels(n_hyb + 1)) << "fermionic flavour " << i << " merged with the S+ set";
    }

    DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, /*n_int=*/2);
    EXPECT_EQ(D.n_hyb, n_hyb);
    EXPECT_EQ(D.n_int, 2);
    EXPECT_EQ(D.n, n_hyb + 2);
  }
}

/**
 * @brief Check that any off-diagonal entry of dynint_coeffs between S- and S+, symmetric or not, is rejected as a cross-set coefficient
 */
TEST(BlockSparseDynintSpinFlip, rejects_off_diagonal_interaction_coefficients_between_S_minus_and_S_plus) {

  auto m = spin_flip_model(/*norb=*/2);

  auto try_build = [&](nda::array<dcomplex, 3> const &d) {
    try {
      auto q = get_operators_and_interactions(m.ad, m.hyb_coeffs, d, m.dynint_ops);
      return std::string{};
    } catch (std::invalid_argument const &e) { return std::string{e.what()}; }
  };

  // the diagonal fixture must be accepted
  EXPECT_EQ(try_build(m.dynint_coeffs), std::string{}) << "the diagonal two-entry encoding must be accepted";

  for (auto const &[name, d01, d10] :
       std::vector<std::tuple<std::string, dcomplex, dcomplex>>{{"asymmetric off-diagonal", 0.31, 0.17}, {"symmetric off-diagonal", 0.31, 0.31}}) {
    SCOPED_TRACE(name);
    auto d     = nda::array<dcomplex, 3>{m.dynint_coeffs};
    d(0, 0, 1) = d01;
    d(0, 1, 0) = d10;
    auto msg   = try_build(d);
    EXPECT_NE(msg.find("couples different symmetry sets"), std::string::npos) << "message was: " << msg;
  }
}

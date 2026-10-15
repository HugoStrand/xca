#include <limits>
#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/dense/fset.hpp>

#include "test_utils/block_sparse.hpp"

using nda::dcomplex;

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;
using triqs_xca::block_sparse::BlockOpSymQuartet;
namespace test_utils = triqs_xca::test_utils;

namespace block_sparse = triqs_xca::block_sparse;

namespace dense = triqs_xca::dense;


/**
 * @file block_sparse/sym_sets.cpp
 *
 * @brief Tests of the barred-operator construction in the BlockOpSymQuartet constructor
 *
 * @details The contraction of the hybridization coefficients with the operators of a symmetry set is only defined within one set, as in the dense
 * reference FSet::update_hybridization(). A hybridization coupling two different symmetry sets can not be represented by the block-sparse
 * storage and must be rejected by the constructor, regardless of coefficient magnitude. With symmetry sets of unequal size, an index
 * mix-up between the sets is an out-of-bounds read that only shows up in a bounds-checked or
 * sanitized build (-DASAN=ON or -DNDA_ENFORCE_BOUNDCHECK), so the barred operators are also compared to the dense reference, which is defined
 * for any coefficient matrix.
 */

/**
 * @brief Check that the model has two symmetry sets of unequal size, so that the tests below are not vacuous
 */
TEST(BlockSparseSymSets, model_has_unequal_symmetry_sets) {

  auto ad = test_utils::unequal_sym_set_model();
  int p   = 2;

  auto zero_coeffs  = nda::zeros<dcomplex>(p, 3, 3);
  auto [Fq, labels] = block_sparse::atom_diag::get_operators(ad, zero_coeffs);

  ASSERT_EQ(labels.size(), 3);
  ASSERT_EQ(nda::max_element(Fq.sym_set_labels) + 1, 2) << "expected exactly two symmetry sets, got labels " << labels;
  ASSERT_EQ(nda::sum(Fq.sym_set_sizes), 3);
  ASSERT_NE(Fq.sym_set_sizes(0), Fq.sym_set_sizes(1)) << "expected symmetry sets of unequal size, got " << Fq.sym_set_sizes;
}

/**
 * @brief Check that a hybridization coupling two symmetry sets is rejected by the BlockOpSymQuartet constructor
 */
TEST(BlockSparseSymSets, bars_reject_symmetry_set_coupling) {

  auto ad = test_utils::unequal_sym_set_model();
  int p   = 2;

  auto labels = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, 3, 3)));

  // Find a pair of orbitals living in different symmetry sets.
  int i_cross = -1, j_cross = -1;
  for (int i = 0; i < labels.size() && i_cross < 0; ++i) {
    for (int j = 0; j < labels.size(); ++j) {
      if (labels(i) != labels(j)) {
        i_cross = i;
        j_cross = j;
        break;
      }
    }
  }
  ASSERT_GE(i_cross, 0);

  auto hyb_coeffs = test_utils::sym_set_diagonal_hyb(labels, p);

  // couple the two symmetry sets
  hyb_coeffs(0, i_cross, j_cross) = 0.42;
  hyb_coeffs(0, j_cross, i_cross) = 0.42;

  // match the message, other std::invalid_argument checks sit earlier in the same constructor
  try {
    block_sparse::atom_diag::get_operators(ad, hyb_coeffs);
    FAIL() << "hyb_coeffs couples orbitals " << i_cross << " and " << j_cross
           << ", which are in different symmetry sets; the BlockOpSymQuartet constructor must reject that";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string(e.what()).find("couples different symmetry sets"), std::string::npos) << e.what();
  }
}

TEST(BlockSparseSymSets, rejects_every_nonzero_cross_set_coefficient) {
  auto ad = test_utils::unequal_sym_set_model();
  auto labels = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(2, 3, 3)));
  int i = -1, j = -1;
  for (int a = 0; a < labels.size(); ++a)
    for (int b = 0; b < labels.size(); ++b)
      if (labels(a) != labels(b)) { i = a; j = b; }
  ASSERT_GE(i, 0);
  for (double scale : {1., 1e6}) {
    auto coefficients = nda::make_regular(scale * test_utils::sym_set_diagonal_hyb(labels, 2));
    EXPECT_NO_THROW(block_sparse::atom_diag::get_operators(ad, coefficients));
    for (double magnitude : {std::numeric_limits<double>::denorm_min(), 1e-18, 1e-13, 1e-6}) {
      SCOPED_TRACE(::testing::Message() << "scale=" << scale << ", magnitude=" << magnitude);
      coefficients(0, i, j) = dcomplex(0., magnitude);
      EXPECT_THROW(block_sparse::atom_diag::get_operators(ad, coefficients), std::invalid_argument);
      coefficients(0, i, j) = 0.;
    }
  }
}

TEST(BlockSparseSymSets, rejects_nonfinite_coefficients_within_and_between_sets) {
  using triqs_xca::block_sparse::check_sym_set_block_diagonal;
  nda::vector<long> labels{0, 1};
  for (int j : {0, 1})
    for (dcomplex value : {dcomplex(std::numeric_limits<double>::quiet_NaN(), 0.),
                           dcomplex(0., std::numeric_limits<double>::quiet_NaN()),
                           dcomplex(std::numeric_limits<double>::infinity(), 0.)}) {
      auto coefficients = nda::zeros<dcomplex>(1, 2, 2);
      coefficients(0, 0, 0) = 1.;
      coefficients(0, 0, j) = value;
      EXPECT_THROW(check_sym_set_block_diagonal(coefficients, labels, "test"), std::invalid_argument);
    }
}

/**
 * @brief Compare the barred operators of symmetry sets of unequal size to the dense reference
 */
TEST(BlockSparseSymSets, bars_match_dense_for_unequal_sym_sets) {

  auto ad = test_utils::unequal_sym_set_model();
  int p   = 2;

  auto labels     = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, 3, 3)));
  auto hyb_coeffs = test_utils::sym_set_diagonal_hyb(labels, p);

  auto [Fq, labels2] = block_sparse::atom_diag::get_operators(ad, hyb_coeffs);
  auto Fset          = dense::atom_diag::get_operators(ad, hyb_coeffs); // dense reference, no symmetry sets involved

  int n_sets    = static_cast<int>(nda::max_element(Fq.sym_set_labels) + 1);
  double da_err = 0.0, fb_err = 0.0;

  for (int q_ix = 0; q_ix < n_sets; ++q_ix) {
    auto const &F_dag_bar  = Fq.F_dag_bars[q_ix];
    auto const &F_bar_refl = Fq.F_bars_refl[q_ix];

    for (int i = 0; i < Fq.sym_set_sizes(q_ix); ++i) {
      long orb = Fq.sym_set_to_orb(q_ix, i);

      for (int b = 0; b < ad.n_subspaces(); ++b) {
        auto const &fock_col = ad.get_fock_states(b);

        int bc_dag = F_dag_bar.get_block_index(b);
        if (bc_dag != -1) {
          auto const &fock_row = ad.get_fock_states(bc_dag);
          for (int l = 0; l < p; ++l) {
            for (int ri = 0; ri < fock_row.size(); ++ri) {
              for (int cj = 0; cj < fock_col.size(); ++cj) {
                auto bs    = F_dag_bar.get_block(b)(i, l, ri, cj);
                auto dense = Fset.F_dag_bars(orb, l, fock_row[ri], fock_col[cj]);
                da_err     = std::max(da_err, std::abs(bs - dense));
              }
            }
          }
        }

        int bc = F_bar_refl.get_block_index(b);
        if (bc != -1) {
          auto const &fock_row = ad.get_fock_states(bc);
          for (int l = 0; l < p; ++l) {
            for (int ri = 0; ri < fock_row.size(); ++ri) {
              for (int cj = 0; cj < fock_col.size(); ++cj) {
                auto bs    = F_bar_refl.get_block(b)(i, l, ri, cj);
                auto dense = Fset.F_bars_refl(orb, l, fock_row[ri], fock_col[cj]);
                fb_err     = std::max(fb_err, std::abs(bs - dense));
              }
            }
          }
        }
      }
    }
  }

  EXPECT_LE(da_err, 1.0e-14) << "block-sparse F_dag_bars disagree with the dense reference";
  EXPECT_LE(fb_err, 1.0e-14) << "block-sparse F_bars_refl disagree with the dense reference";
}

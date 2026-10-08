#include <gtest/gtest.h>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/atom_diag.hpp>

#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/hyb.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

namespace dense = triqs_xca::dense;

using triqs_xca::block_sparse::DiagramEvaluator;
using triqs_xca::block_sparse::BlockDiagOpFun;
namespace test_utils = triqs_xca::test_utils;


/**
 * @file block_sparse/evaluator_api.cpp
 *
 * @brief Tests that the several ways of calling a DiagramEvaluator all describe the same problem
 *
 * @details Diagram evaluators can be constructed in two ways in C++, either from an atom_diag object, which derives the field operators and the 
 * block structure itself and takes the propagator as a block_gf, or from a BlockOpSymQuartet assembled by the caller, which takes the propagator as 
 * a BlockDiagOpFun. The flat-index overloads, through which the caller drives the backbone loop, are checked for every evaluator in
 * flat_index_all_evals.cpp.
 *
 * These are consistency checks between code paths rather than checks of the diagram itself: the routes build the same Hybridization from the same
 * coefficients and sum the same backbones, so they should agree to machine precision.
 *
 * Every test here uses the same model, the two-band Kanamori atom with a two-pole discrete-bath hybridization, and takes its atom_diag from
 * two_band_atom_diag_helper, so the subspaces, and hence the block ordering, are identical throughout.
 */

/**
 * @brief Check that the two DiagramEvaluator constructors give the same self-energy
 *
 * @details The BlockOpSymQuartet constructor is C2PY_IGNORE and predates the atom_diag one, but is still used internally, so the two must agree on
 * the same model.
 */
TEST(EvaluatorAPI, OCA_constructor_equivalence) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = test_utils::discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  // --- atom_diag constructor, taking the atomic propagator as a block_gf ---
  auto ad      = test_utils::two_band_atom_diag_helper();
  auto G0_ppsc = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  BlockDiagOpFun OCA_result(D.compute_self_energy(G0_ppsc, topology));

  // --- old constructor, taking field operators as a BlockOpSymQuartet and the propagator as a BlockDiagOpFun ---
  auto [Gt, Fq, sym_set_labels] = test_utils::two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator DDE(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  BlockDiagOpFun OCA_result_2(DDE.compute_self_energy(Gt, topology));

  // the two constructors must assemble the same hybridization ...
  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - DDE.hyb.values)), 1.0e-15);
  ASSERT_EQ(D.hyb.poles, DDE.hyb.poles);

  // ... and hence the same self-energy, block by block
  ASSERT_EQ(OCA_result.get_num_block_cols(), OCA_result_2.get_num_block_cols());
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {
    SCOPED_TRACE("block " + std::to_string(i));
    EXPECT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - OCA_result_2.get_block(i))), 1.0e-15);
  }
}

/**
 * @brief Check that the two DiagramEvaluator constructors give the same single-particle Green's function
 *
 * @details The single-particle analogue of OCA_constructor_equivalence above, with a comparison against the dense evaluator appended: unlike the
 * self-energy, whose block-sparse-vs-dense comparison is made in test_block_sparse_sparsity_invariance.cpp, the single-particle Green's function is
 * a dense object in orbital space and is compared directly.
 */
TEST(EvaluatorAPI, gf_constructor_equivalence) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = test_utils::discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // Kanamori atom and its atomic propagator as a block_gf
  auto ad      = test_utils::two_band_atom_diag_helper();
  auto G0_ppsc = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  // --- atom_diag constructor, taking the atomic propagator as a block_gf ---
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf = D.compute_single_ptcle_gf(G0_ppsc, topology);

  // --- old constructor, taking field operators as a BlockOpSymQuartet and the propagator as a BlockDiagOpFun ---
  auto [Gt, Fq, sym_set_labels] = test_utils::two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D2(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto OCA_gf_2 = D2.compute_single_ptcle_gf(Gt, topology);

  // the two constructors must assemble the same hybridization ...
  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - D2.hyb.values)), 1.0e-15);
  ASSERT_EQ(D.hyb.poles, D2.hyb.poles);

  // ... and hence the same Green's function
  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_2)), 1.0e-15);

  // --- dense evaluation, over the full Hilbert space as a single block ---
  auto G0t_dense = test_utils::Hmat_to_Gtmat(test_utils::get_full_h_atomic(ad), beta, cppdlr::rel2abs(itops.get_itnodes()));
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), G0t_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  dense::DiagramEvaluator D_dense(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf_dense = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology);
  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_dense)), 1.0e-15);
}

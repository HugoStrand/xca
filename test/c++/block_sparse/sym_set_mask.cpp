#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/topology.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>

#include "test_utils/block_sparse.hpp"

using nda::dcomplex;

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::block_sparse::atom_diag::get_operators;

using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOpSymQuartet;
using triqs_xca::block_sparse::DiagramEvaluator;
namespace test_utils = triqs_xca::test_utils;

/**
 * @file block_sparse/sym_set_mask.cpp
 *
 * @brief Tests of the symmetry-set mask on the zero vertex of the self-energy, and of the input validation of the Fq-only constructor
 *
 * @details A nonzero cross-set coefficient must be represented or rejected explicitly. The guard is structural,
 * not an approximation based on coefficient magnitude. The dense/coarse-block reference below demonstrates that
 * a tiny coefficient can generate an off-block self-energy which a fine block-diagonal layout cannot store.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  constexpr int p_poles = 2;

  /// Where the perturbation goes: the cross-set entries, or the within-set off-diagonals used as the control
  enum class Slot { cross, within };

  /// The symmetry-set labels the quartet assigns to this model's flavours.
  nda::vector<int> sym_set_labels_of(triqs::atom_diag::atom_diag<true> const &ad) {
    int n_hyb   = static_cast<int>(ad.get_fops().size());
    auto labels = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
    nda::vector<int> lab(labels.size());
    for (long i = 0; i < labels.size(); ++i) lab(i) = static_cast<int>(labels(i));
    return lab;
  }

  /**
   * @brief sym_set_diagonal_hyb with every entry of slot shifted by +-delta, with alternating sign so that the perturbations sum to zero
   */
  nda::array<dcomplex, 3> perturbed_hyb(nda::vector_const_view<int> lab, Slot slot, double delta) {
    int n_hyb  = static_cast<int>(lab.size());
    auto c     = nda::array<dcomplex, 3>(test_utils::sym_set_diagonal_hyb(lab, p_poles));
    double sgn = 1.0;
    for (int l = 0; l < p_poles; ++l)
      for (int i = 0; i < n_hyb; ++i)
        for (int j = 0; j < n_hyb; ++j) {
          bool hit = (slot == Slot::cross) ? (lab(i) != lab(j)) : (lab(i) == lab(j) && i != j);
          if (hit) {
            c(l, i, j) += sgn * delta;
            sgn = -sgn;
          }
        }
    return c;
  }

  /// Sigma at the crossing OCA topology, from coefficients the quartet is also built from
  BlockDiagOpFun sigma_from(triqs::atom_diag::atom_diag<true> const &ad, nda::array<dcomplex, 3> const &coeffs) {
    nda::vector<double> hyb_poles = {1.3, -0.8};
    nda::array<int, 2> topology   = {{0, 2}, {1, 3}};

    auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
    auto Fq    = std::get<0>(get_operators(ad, coeffs));
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, coeffs, Fq, /*n_int=*/0);
    auto Gt = test_utils::ad_to_atom_prop(ad, beta, itops);
    return BlockDiagOpFun(D.compute_self_energy(Gt, topology));
  }

  double max_block_diff(BlockDiagOpFun const &A, BlockDiagOpFun const &B) {
    double d = 0.0;
    for (int b = 0; b < A.get_num_block_cols(); ++b) d = std::max(d, nda::max_element(nda::abs(A.get_block(b) - B.get_block(b))));
    return d;
  }

  double max_block_abs(BlockDiagOpFun const &A) {
    double d = 0.0;
    for (int b = 0; b < A.get_num_block_cols(); ++b) d = std::max(d, nda::max_element(nda::abs(A.get_block(b))));
    return d;
  }

} // namespace

TEST(BlockSparseSymSetMask, tiny_cross_set_coefficients_are_rejected_but_within_set_terms_survive) {
  for (auto const &ad : {test_utils::unequal_sym_set_model(true), test_utils::spin_flip_atom_diag_helper(2, false),
                        test_utils::sz_resolved_atom_diag_helper(2, true)}) {
    auto labels = sym_set_labels_of(ad);
    SCOPED_TRACE(::testing::Message() << "labels=" << labels);
    ASSERT_GE(nda::max_element(labels), 1);
    auto clean = perturbed_hyb(labels, Slot::cross, 0.);
    double delta = 0.9e-12 * nda::max_element(nda::abs(clean)); // formerly accepted
    EXPECT_THROW(get_operators(ad, perturbed_hyb(labels, Slot::cross, delta)), std::invalid_argument);
    auto baseline = sigma_from(ad, clean);
    auto within = sigma_from(ad, perturbed_hyb(labels, Slot::within, delta));
    ASSERT_GT(max_block_abs(baseline), 0.1);
    EXPECT_GT(max_block_diff(within, baseline), 0.); // no threshold on representable terms
  }
}

TEST(BlockSparseSymSetMask, discarded_remainder_has_a_resolvable_off_block_self_energy) {
  using tensor = nda::array<dcomplex, 3>;
  using gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>;
  using bgf = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  // Fock basis |00>, |10>, |01>, |11>, with c1|11> = -|10>.
  tensor F = nda::zeros<dcomplex>(2, 4, 4), Fdag = F;
  F(0, 0, 1) = F(0, 2, 3) = F(1, 0, 2) = 1.;
  F(1, 1, 3) = -1.;
  for (int a = 0; a < 2; ++a)
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) Fdag(a, i, j) = F(a, j, i);
  auto mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, 1e-12, false);
  auto itops = mesh.dlr_it();
  nda::vector<double> poles{0.};
  nda::array<int, 2> topology{{0, 1}};
  nda::vector<int> connection{0};
  std::vector<tensor> blocks{F}, dag_blocks{Fdag};
  triqs_xca::block_sparse::BlockOpSymSet grouped_F(connection, blocks), grouped_Fdag(connection, dag_blocks);
  tensor coefficients = nda::zeros<dcomplex>(1, 2, 2);
  coefficients(0, 0, 0) = coefficients(0, 1, 1) = 1.;
  double delta = 1e-14;
  coefficients(0, 0, 1) = coefficients(0, 1, 0) = delta;
  BlockOpSymQuartet quartet({grouped_F}, {grouped_Fdag}, coefficients, nda::vector<long>{0, 0});
  DiagramEvaluator coarse(beta, Lambda, 1e-12, poles, coefficients, quartet);
  triqs_xca::dense::FSet fset(F, Fdag, coefficients);
  triqs_xca::dense::DiagramEvaluator dense(beta, 1e-12, itops, poles, coefficients, fset);

  for (double scale : {1., 1e8}) {
    SCOPED_TRACE(::testing::Message() << "propagator scale=" << scale);
    tensor values = nda::zeros<dcomplex>(mesh.size(), 4, 4);
    for (int t = 0; t < mesh.size(); ++t) {
      values(t, 0, 0) = scale;
      values(t, 1, 1) = 2 * scale;
      values(t, 2, 2) = 4 * scale;
      values(t, 3, 3) = 3 * scale;
    }
    bgf G{std::vector{gf(mesh, values)}};
    auto dense_sigma = dense.compute_self_energy(G, topology);
    auto coarse_sigma = coarse.compute_self_energy(G, topology);
    // Delta=-C/2. At matrix entry (1,2), c0^dag G c1 contributes G00,
    // c1 G c0^dag contributes -G33: Sigma12=-delta/2*(G00-G33)=delta*scale.
    // This entry is absent in a four-sector block-diagonal representation.
    double expected = delta * scale;
    for (int t = 0; t < mesh.size(); ++t) {
      EXPECT_LE(std::abs(dense_sigma[0].data()(t, 1, 2) - expected), 2e-11 * expected);
      EXPECT_LE(std::abs(coarse_sigma[0].data()(t, 1, 2) - expected), 2e-11 * expected);
    }
  }
}

/**
 * @brief Check that the Fq-only constructor rejects coefficients that couple different symmetry sets
 *
 * @details The atom_diag constructors build the quartet and the hybridization from the same array, while the Fq-only constructor takes them
 * separately and the quartet stores no coefficients to check against.
 */
TEST(BlockSparseSymSetMask, Fq_only_constructor_rejects_cross_set_coefficients) {
  auto ad   = test_utils::unequal_sym_set_model(true);
  auto lab  = sym_set_labels_of(ad);
  auto good = perturbed_hyb(lab, Slot::cross, 0.0);
  auto Fq   = std::get<0>(get_operators(ad, good));

  nda::vector<double> hyb_poles = {1.3, -0.8};
  ASSERT_NO_THROW(DiagramEvaluator(beta, Lambda, eps, hyb_poles, good, Fq)) << "the clean pair must still construct";

  // Even a subnormal cross-set coefficient must be rejected by both entry points.
  auto bad = perturbed_hyb(lab, Slot::cross, std::numeric_limits<double>::denorm_min());
  ASSERT_THROW(std::get<0>(get_operators(ad, bad)), std::invalid_argument) << "premise: the quartet rejects this array";

  try {
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, bad, Fq);
    FAIL() << "the Fq-only constructor accepted hyb_coeffs that couple different symmetry sets, which the "
              "quartet it was handed cannot represent";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string(e.what()).find("symmetry set"), std::string::npos)
       << "wrong throw - the message should name the symmetry-set coupling: " << e.what();
  }
}

/**
 * @brief Check that the Fq-only constructor rejects coefficients whose extents disagree with the quartet or the poles
 */
TEST(BlockSparseSymSetMask, Fq_only_constructor_rejects_mismatched_coefficient_extents) {
  auto ad  = test_utils::unequal_sym_set_model(true);
  auto lab = sym_set_labels_of(ad);
  auto Fq  = std::get<0>(get_operators(ad, perturbed_hyb(lab, Slot::cross, 0.0)));

  int n = static_cast<int>(nda::sum(Fq.sym_set_sizes));
  ASSERT_GT(n, 2) << "vacuous test: the truncated array below must be smaller than the quartet";

  auto too_small                = nda::zeros<dcomplex>(p_poles, n - 1, n - 1);
  nda::vector<double> hyb_poles = {1.3, -0.8};

  // match on "shape": with the extent check deleted, check_sym_set_block_diagonal would read past the smaller array and throw the
  // cross-set message instead
  try {
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, too_small, Fq);
    FAIL() << "the Fq-only constructor accepted a (" << p_poles << ", " << n - 1 << ", " << n - 1 << ") coefficient array for a quartet holding " << n
           << " orbitals";
  } catch (std::invalid_argument const &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("shape"), std::string::npos)
       << "wrong throw - this must be the extent check, not the cross-set check reading past the array: " << msg;
    EXPECT_NE(msg.find("sum(Fq.sym_set_sizes)"), std::string::npos) << "the message should say what n was compared against: " << msg;
  }

  // the pole extent too, a wrong p is silently reinterpreted by the reshape in coefs2vals
  auto wrong_poles = nda::zeros<dcomplex>(p_poles + 1, n, n);
  try {
    DiagramEvaluator D(beta, Lambda, eps, hyb_poles, wrong_poles, Fq);
    FAIL() << "the Fq-only constructor accepted a (" << p_poles + 1 << ", " << n << ", " << n << ") coefficient array against " << hyb_poles.size()
           << " hybridization poles";
  } catch (std::invalid_argument const &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("shape"), std::string::npos) << "wrong throw - this must be the extent check: " << msg;
    EXPECT_NE(msg.find("hyb_poles.size()"), std::string::npos) << "the message should name hyb_poles as the thing that disagrees: " << msg;
  }
}

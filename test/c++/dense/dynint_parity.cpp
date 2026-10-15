#include <stdexcept>

#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/diagram/topology.hpp>
#include "test_utils/block_sparse.hpp"

using nda::dcomplex;

using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

using triqs_xca::dense::DiagramEvaluator;
namespace test_utils = triqs_xca::test_utils;

/**
 * @file dense/dynint_parity.cpp
 *
 * @brief Regression tests of the dense diagram evaluator with dynamical interactions
 *
 * @details Permutation signs use explicit interaction statistics from the model catalogue.
 * Both plain and paired evaluation must classify retarded interactions as bosonic.
 * Only a crossing topology exposes an incorrect all-fermionic classification, so
 * the tests use the order-2 topology {{0,2},{1,3}}, which is odd while all of its sub-matchings are even. The flat-index overloads of the
 * self-energy and the single-particle Green's function are checked for every evaluator in flat_index_all_evals.cpp; the by-pairs
 * self-energy, which only the dense evaluator has, is checked here. The tests of the single-subspace requirement cover both the dense
 * dynamical interaction operator construction and compute_one_time_correlator(), which read the operator matrices from subspace 0 only.
 */

namespace {

  struct Model {
    triqs::atom_diag::atom_diag<true> ad;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_ppsc;
    nda::vector<double> hyb_poles;
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> dynint_coeffs;
    std::vector<many_body_operator_complex> dynint_ops;
  };

  /**
   * @brief Single spinless level with a retarded interaction D(tau) coupled to the density
   *
   * @details With partition = false the atom_diag has a single subspace, as required by the dense dynamical interaction path. With partition = true
   * the particle number is conserved and the atom_diag has two subspaces of dimension 1.
   */
  Model dynint_model(double beta, double Lambda, double eps, bool partition = false) {

    many_body_operator_complex H = -0.3 * n("0", 0);

    triqs::atom_diag::fundamental_operator_set fop_set;
    fop_set.insert("0", 0);

    many_body_operator_complex Nop;
    Nop = n("0", 0);

    std::vector<many_body_operator_complex> sym_ops = {}; // empty -> one subspace
    if (partition) sym_ops.push_back(Nop);                // conserving N -> two subspaces of dim 1

    auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set, sym_ops);

    auto G_ppsc = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);

    // Two-pole representation shared by the hybridization and the retarded interaction,
    // as produced by the joint adapol/DLR fit in the solver.
    int p = 2;
    nda::vector<double> hyb_poles(p);
    hyb_poles(0) = 1.3;
    hyb_poles(1) = -0.8;

    auto hyb_coeffs = nda::zeros<dcomplex>(p, 1, 1);
    hyb_coeffs(0, 0, 0) = 0.7;
    hyb_coeffs(1, 0, 0) = 0.5;

    auto dynint_coeffs = nda::zeros<dcomplex>(p, 1, 1);
    dynint_coeffs(0, 0, 0) = 0.9;
    dynint_coeffs(1, 0, 0) = 0.6;

    std::vector<many_body_operator_complex> dynint_ops = {n<double>("0", 0)};

    return {.ad            = ad,
            .G_ppsc        = G_ppsc,
            .hyb_poles     = hyb_poles,
            .hyb_coeffs    = hyb_coeffs,
            .dynint_coeffs = dynint_coeffs,
            .dynint_ops    = dynint_ops};
  }

} // namespace

/**
 * @brief Check that compute_self_energy_by_pairs() agrees with compute_self_energy() in the presence of dynamical interactions
 *
 * @details The by-pairs routine evaluates both directions of the hybridization line attached to vertex 0 in one pass, and must use the same
 * interaction statistics as the plain routine.
 */
TEST(DenseDynint, self_energy_by_pairs_agrees) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}}; // crossing: the only order-2 topology that flips

  // only a crossing topology exposes the parity error
  ASSERT_EQ(triqs_xca::diagram::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  ASSERT_EQ(D.n_hyb, 1);
  ASSERT_EQ(D.n_int, 1);
  ASSERT_EQ(D.n, 2);

  // check that some backbones carry the interaction operator on the internal line
  DiagramEvaluator D_no_dynint(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad);

  long n_backbones           = D.get_num_self_energy_backbones(topology);
  long n_backbones_fermionic = D_no_dynint.get_num_self_energy_backbones(topology);
  ASSERT_GT(n_backbones, n_backbones_fermionic) << "vacuous test: no backbone carries the interaction operator";

  auto sigma_all   = D.compute_self_energy(m.G_ppsc, topology);
  auto sigma_pairs = D.compute_self_energy_by_pairs(m.G_ppsc, topology);

  // the comparison below is absolute, so require a non-negligible self-energy
  double scale = nda::max_element(nda::abs(sigma_all[0].data()));
  ASSERT_GT(scale, 0.1) << "vacuous test: the self-energy is too small for an absolute tolerance";

  double err = nda::max_element(nda::abs(sigma_all[0].data() - sigma_pairs[0].data()));
  if (err > 1.0e-12) std::cout << "max|Sigma| = " << scale << ", max|all - by_pairs| = " << err << "\n";

  EXPECT_LE(err, 1.0e-12) << "compute_self_energy_by_pairs(G, topology) disagrees with compute_self_energy(G, topology)";
}

/**
 * @brief Check that the three compute_self_energy_by_pairs() overloads agree
 *
 * @details The flat-index overloads do not filter the flat indices, so the caller has to pass only the indices with fb(0) = 0, i.e.
 * (f_ix / n_p) % 2 == 0 with n_p = o_ix_max * p_ix_max, as Solver.eval_pseudo_particle_self_energy_order_by_pairs does.
 */
TEST(DenseDynint, self_energy_by_pairs_flat_index_overloads_agree) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  ASSERT_EQ(triqs_xca::diagram::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  // stride n_p of fb_ix in the flat index, from f_ix_max = n_p * fb_ix_max with fb_ix_max = 2^m for self-energy assignments
  int order    = topology.extent(0);
  long f_ix_max = D.get_num_self_energy_backbones(topology);
  int n_p      = f_ix_max / (1 << order);

  std::vector<int> seeds;
  for (int f_ix = 0; f_ix < f_ix_max; ++f_ix) {
    if ((f_ix / n_p) % 2 == 0) seeds.push_back(f_ix);
  }
  ASSERT_EQ(static_cast<long>(seeds.size()), f_ix_max / 2);

  auto sigma_all = D.compute_self_energy_by_pairs(m.G_ppsc, topology);
  ASSERT_GT(nda::max_element(nda::abs(sigma_all[0].data())), 1.0e-8) << "vacuous test: the self-energy is zero";

  nda::vector<long> f_ix_vec(seeds.size());
  for (long i = 0; i < static_cast<long>(seeds.size()); ++i) f_ix_vec(i) = seeds[i];
  auto sigma_vec = D.compute_self_energy_by_pairs(m.G_ppsc, topology, f_ix_vec);

  auto sigma_single = nda::zeros<dcomplex>(D.r, D.N, D.N);
  for (int f_ix : seeds) sigma_single += D.compute_self_energy_by_pairs(m.G_ppsc, topology, f_ix)[0].data();

  EXPECT_LE(nda::max_element(nda::abs(sigma_all[0].data() - sigma_vec[0].data())), 1.0e-14)
     << "compute_self_energy_by_pairs(G, topology, f_ix_vec) disagrees with compute_self_energy_by_pairs(G, topology)";

  EXPECT_LE(nda::max_element(nda::abs(sigma_all[0].data() - sigma_single)), 1.0e-14)
     << "compute_self_energy_by_pairs(G, topology, f_ix) disagrees with compute_self_energy_by_pairs(G, topology)";
}

/**
 * @brief Check that the dense dynamical interaction constructor rejects an atom_diag with more than one subspace
 *
 * @details The operator matrices are read from subspace 0 only, and with more than one subspace the assignment to the full Hilbert space
 * matrix reads past the end of the source in a build without bounds checks.
 */
TEST(DenseDynint, dynint_constructor_rejects_multi_subspace_atom_diag) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps, /*partition=*/true);

  ASSERT_EQ(m.ad.n_subspaces(), 2);
  ASSERT_EQ(m.ad.get_full_hilbert_space_dim(), 2);

  // match the message, std::invalid_argument is also thrown upstream of the check under test
  try {
    DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
    (void)D;
    FAIL() << "expected std::invalid_argument for a multi-subspace atom_diag";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("single subspace"), std::string::npos) << e.what(); }
}

/**
 * @brief Check that compute_one_time_correlator() rejects an atom_diag with more than one subspace
 */
TEST(DenseDynint, one_time_correlator_rejects_multi_subspace_atom_diag) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps, /*partition=*/true);

  ASSERT_EQ(m.ad.n_subspaces(), 2);

  // the constructor without dynamical interactions handles several subspaces, the correlator routine does not
  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad);

  nda::array<int, 2> topology              = {{0, 1}};
  std::vector<many_body_operator_complex> ops = {n<double>("0", 0)};

  nda::vector<long> f_ix_vec(1);
  f_ix_vec(0) = 0;

  // a propagator in the layout the dense evaluator takes, so that only the atom_diag is at fault
  auto G_flat = dynint_model(beta, Lambda, eps).G_ppsc;

  try {
    D.compute_one_time_correlator(G_flat, ops, ops, m.ad, topology, f_ix_vec);
    FAIL() << "expected std::invalid_argument for a multi-subspace atom_diag";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("single subspace"), std::string::npos) << e.what(); }
}

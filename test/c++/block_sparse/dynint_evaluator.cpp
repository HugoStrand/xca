#include <algorithm>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>
#include <triqs_xca/topology.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense_comparison.hpp"
#include "test_utils/parallel_atom_diag_check.hpp"

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;
using nda::dcomplex;

using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

using triqs_xca::block_sparse::atom_diag::get_operators;
using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOpSymQuartet;
using triqs_xca::block_sparse::DiagramEvaluator;
namespace dense = triqs_xca::dense;
using triqs_xca::hyb::get_extended_coefficients;
using triqs_xca::block_sparse::dynint::get_operators_and_interactions;
namespace test_utils = triqs_xca::test_utils;

/**
 * @file block_sparse/dynint_evaluator.cpp
 *
 * @brief Tests of the block-sparse DiagramEvaluator with dynamical interactions: the constructors, n / n_hyb / n_int and Nmax
 *
 * @details The comparisons with the dense evaluator in this file drive the internal constructor that takes the field operators; the dense
 * evaluator runs on the single-subspace twin of the partitioned atom_diag, with the two eigenbases bridged by
 * get_tensor_in_atom_diag_subspace(). The self-energy and the single-particle Green's function of the public constructors are compared
 * across evaluators in sigma_spgf_all_evals.cpp and backbone by backbone in flat_index_all_evals.cpp; the one-time correlator is compared
 * across evaluators in one_time_correlator_all_evals.cpp, and that of the evaluator built from the quartet in dynint_constructor_contract.
 */

namespace {

  // tolerance for the comparison with the dense reference, the measured agreement is ~1e-16
  constexpr double sigma_tol = 1.0e-13;

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  constexpr int p_poles = 2;

  struct DynintModel {
    triqs::atom_diag::atom_diag<true> ad;
    nda::vector<double> hyb_poles;
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> dynint_coeffs;
    std::vector<many_body_operator_complex> dynint_ops;
  };

  /**
   * @brief unequal_sym_set_model plus n_int density interactions
   *
   * @details The symmetry sets have unequal sizes {2, 1} and the largest subspace has dimension 2, so the Nmax tests are not vacuous.
   */
  DynintModel dynint_model(int n_int) {
    auto ad       = test_utils::unequal_sym_set_model(true);
    int n_hyb     = static_cast<int>(ad.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
    auto hyb      = test_utils::sym_set_diagonal_hyb(labels_f, p_poles);

    std::vector<many_body_operator_complex> ops;
    if (n_int > 0) ops.emplace_back(n("A", 0));
    // n_A0 and n_B0 have different support but compatible diagonal group connectivity
    if (n_int > 1) ops.emplace_back(n("B", 0));
    EXPECT_EQ(static_cast<int>(ops.size()), n_int) << "dynint_model only knows two interaction operators";

    // l- and i-dependent, so that a wrong pole or interaction flavour shows up
    auto d = nda::zeros<dcomplex>(p_poles, n_int, n_int);
    for (int l = 0; l < p_poles; ++l)
      for (int i = 0; i < n_int; ++i) d(l, i, i) = 0.61 - 0.07 * l + 0.05 * i;

    nda::vector<double> hyb_poles = {1.3, -0.8};
    return {ad, hyb_poles, hyb, d, ops};
  }

  /// Same quartet with symmetry set s moved to position 0
  BlockOpSymQuartet permute_set_to_front(BlockOpSymQuartet const &Fq, long s, nda::array_const_view<dcomplex, 3> coeffs) {
    long nsets = nda::max_element(Fq.sym_set_labels) + 1;
    std::vector<long> order{s};
    for (long i = 0; i < nsets; ++i)
      if (i != s) order.push_back(i);
    std::vector<long> pos(nsets);
    for (long i = 0; i < nsets; ++i) pos[order[i]] = i;

    std::vector<triqs_xca::block_sparse::BlockOpSymSet> Fs, F_dags;
    for (long i = 0; i < nsets; ++i) {
      Fs.push_back(Fq.Fs[order[i]]);
      F_dags.push_back(Fq.F_dags[order[i]]);
    }
    auto labels = nda::vector<long>(Fq.sym_set_labels.size());
    for (long i = 0; i < Fq.sym_set_labels.size(); ++i) labels(i) = pos[Fq.sym_set_labels(i)];
    return BlockOpSymQuartet(Fs, F_dags, coeffs, labels);
  }

  int max_subspace_dim(triqs::atom_diag::atom_diag<true> const &ad) {
    auto d = ad.get_subspace_dims();
    return *std::max_element(d.begin(), d.end());
  }

} // namespace

/**
 * @brief Check the atom_diag constructor with dynamical interactions
 *
 * @details hyb_coeffs must cover every fundamental operator of the atom_diag, which is stricter than get_operators().
 */
TEST(BlockSparseDynintEvaluator, dynint_constructor_contract) {

  auto m    = dynint_model(/*n_int=*/2); // two interaction flavours with different exact support
  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());

  auto G = test_utils::ad_to_atom_prop(m.ad, beta, Lambda, eps);
  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  EXPECT_EQ(D.n_hyb, n_hyb);
  EXPECT_EQ(D.n_int, n_int);
  EXPECT_EQ(D.n, n_hyb + n_int);
  ASSERT_GT(n_int, 0) << "vacuous test: no interaction operator";

  // the evaluator built from the quartet must have the same flavours and give the same Sigma and spgf exactly
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto Fq                     = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext                    = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  EXPECT_EQ(D.q, static_cast<int>(nda::max_element(Fq.sym_set_labels) + 1));

  DiagramEvaluator D_fq(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  EXPECT_EQ(D_fq.n_hyb, n_hyb);
  EXPECT_EQ(D_fq.n_int, n_int);
  EXPECT_EQ(D_fq.n, n_hyb + n_int);
  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = test_utils::ad_to_atom_prop(m.ad, beta, itops);

  auto Sa = BlockDiagOpFun(D.compute_self_energy(Gt, topology));
  auto Sb = BlockDiagOpFun(D_fq.compute_self_energy(Gt, topology));
  for (int b = 0; b < Sa.get_num_block_cols(); ++b) EXPECT_EQ(nda::max_element(nda::abs(Sa.get_block(b) - Sb.get_block(b))), 0.0) << "block " << b;

  auto ga = D.compute_single_ptcle_gf(Gt, topology);
  auto gb = D_fq.compute_single_ptcle_gf(Gt, topology);
  ASSERT_EQ(gb.shape(), ga.shape());
  ASSERT_GT(nda::max_element(nda::abs(ga)), 0.01) << "vacuous test: the spgf is zero";
  EXPECT_EQ(nda::max_element(nda::abs(ga - gb)), 0.0);

  // the same one-time correlator exactly, for fermionic legs, whose sign on this topology depends on the statistics of the interaction lines,
  // and for the interaction operators
  nda::vector<long> f_ix_vec(D.get_num_single_ptcle_gf_backbones(topology));
  for (long i = 0; i < f_ix_vec.size(); ++i) f_ix_vec(i) = i;
  std::vector<std::pair<std::vector<many_body_operator_complex>, std::vector<many_body_operator_complex>>> const corr_ops{
     {{triqs::operators::c("A", 0)}, {triqs::operators::c_dag("A", 0)}}, {m.dynint_ops, m.dynint_ops}};
  for (auto const &[ops_tau, ops_0] : corr_ops) {
    auto ca = D.compute_one_time_correlator(G, ops_tau, ops_0, m.ad, topology, f_ix_vec);
    auto cb = D_fq.compute_one_time_correlator(G, ops_tau, ops_0, m.ad, topology, f_ix_vec);
    ASSERT_EQ(cb.shape(), ca.shape());
    ASSERT_GT(nda::max_element(nda::abs(ca)), 0.01) << "vacuous test: the correlator is zero";
    EXPECT_EQ(nda::max_element(nda::abs(ca - cb)), 0.0) << "one-time correlator of " << ops_tau.size() << " operators";
  }

  // match the message, other std::invalid_argument checks sit earlier in the same routine
  auto short_hyb = nda::zeros<dcomplex>(p_poles, n_hyb - 1, n_hyb - 1);
  try {
    DiagramEvaluator D_bad(m.hyb_poles, short_hyb, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
    (void)D_bad;
    FAIL() << "expected std::invalid_argument for a hyb_coeffs that does not cover every fundamental operator";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("fundamental operators"), std::string::npos) << e.what(); }
}

/**
 * @brief Check that the constructor with empty dynint_ops reproduces the plain constructor exactly
 *
 * @details The empty coefficient array has to have shape (p, 0, 0), a default-constructed (0, 0, 0) array is rejected by the shared-pole-count
 * check.
 */
TEST(BlockSparseDynintEvaluator, empty_dynint_ops_reproduce_the_plain_constructor) {

  auto m                      = dynint_model(/*n_int=*/0);
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto G                      = test_utils::ad_to_atom_prop(m.ad, beta, Lambda, eps);

  DiagramEvaluator D0(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad);
  DiagramEvaluator D1(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, {}, nda::zeros<dcomplex>(p_poles, 0, 0));

  EXPECT_EQ(D1.n, D0.n);
  EXPECT_EQ(D1.n_hyb, D0.n_hyb);
  EXPECT_EQ(D1.n_int, 0);
  EXPECT_EQ(D1.q, D0.q);
  EXPECT_EQ(D1.Nmax, D0.Nmax);

  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = test_utils::ad_to_atom_prop(m.ad, beta, itops);
  auto S0    = BlockDiagOpFun(D0.compute_self_energy(Gt, topology));
  auto S1    = BlockDiagOpFun(D1.compute_self_energy(Gt, topology));

  double scale = 0.0;
  for (int b = 0; b < S0.get_num_block_cols(); ++b) {
    scale = std::max(scale, nda::max_element(nda::abs(S0.get_block(b))));
    EXPECT_EQ(nda::max_element(nda::abs(S0.get_block(b) - S1.get_block(b))), 0.0)
       << "block " << b << ": with no interaction operators the two constructors must be bit-identical";
  }
  ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is zero"; // measured 0.134
}

/**
 * @brief Check that the single-particle Green's function has shape (r, n_hyb, n_hyb), i.e. the interaction flavours are not external legs
 */
TEST(BlockSparseDynintEvaluator, spgf_excludes_the_interaction_flavours) {

  auto m    = dynint_model(/*n_int=*/2);
  int n_hyb = static_cast<int>(m.ad.get_fops().size());
  int n_int = static_cast<int>(m.dynint_ops.size());
  ASSERT_GT(n_int, 0) << "vacuous test: without interaction flavours n_hyb == n";

  auto G     = test_utils::ad_to_atom_prop(m.ad, beta, Lambda, eps);
  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = test_utils::ad_to_atom_prop(m.ad, beta, itops);

  DiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
  ASSERT_EQ(D.n, n_hyb + n_int);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto spgf                   = D.compute_single_ptcle_gf(Gt, topology);

  EXPECT_EQ(spgf.extent(1), n_hyb) << "the spgf has " << spgf.extent(1) << " rows, i.e. the interaction flavours are being emitted as external legs";
  EXPECT_EQ(spgf.extent(2), n_hyb) << "the spgf has " << spgf.extent(2)
                                   << " columns, i.e. the interaction flavours are being emitted as external legs";
  EXPECT_EQ(spgf.extent(0), D.r);
}

/**
 * @brief Check that Nmax covers the largest atom_diag subspace, which sizes the evaluation buffers
 */
TEST(BlockSparseDynintEvaluator, Nmax_covers_the_largest_subspace) {

  auto m    = dynint_model(/*n_int=*/1);
  int n_int = static_cast<int>(m.dynint_ops.size());
  int msd   = max_subspace_dim(m.ad);
  ASSERT_GT(msd, 1) << "vacuous test: every subspace of this fixture is one-dimensional"; // measured 2

  auto G = test_utils::ad_to_atom_prop(m.ad, beta, Lambda, eps);
  DiagramEvaluator D_ad(m.hyb_poles, m.hyb_coeffs, G[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);
  EXPECT_GE(D_ad.Nmax, msd);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D_fq(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  EXPECT_GE(D_fq.Nmax, msd);
}

/**
 * @brief Check that Nmax is derived from the whole quartet and not from symmetry set 0
 *
 * @details The symmetry set order is arbitrary, so set 0 may be a projector-like interaction operator with a single 1x1 block. Only the
 * constructor runs here, since evaluating with an undersized Nmax would write out of bounds.
 */
TEST(BlockSparseDynintEvaluator, Nmax_does_not_assume_symmetry_set_zero_covers_the_space) {

  auto ad       = test_utils::unequal_sym_set_model(true);
  int n_hyb     = static_cast<int>(ad.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
  auto hyb      = test_utils::sym_set_diagonal_hyb(labels_f, p_poles);

  // A projector onto a single Fock state: block diagonal, and its only block is 1x1.
  std::vector<many_body_operator_complex> ops = {many_body_operator_complex(n("A", 0) * n("A", 1) * n("B", 0))};
  auto d                                   = nda::zeros<dcomplex>(p_poles, 1, 1);
  d(0, 0, 0)                               = 0.61;
  d(1, 0, 0)                               = 0.54;

  auto Fq  = std::get<0>(get_operators_and_interactions(ad, hyb, d, ops));
  auto ext = get_extended_coefficients(hyb, d);

  long s_int = Fq.sym_set_labels(Fq.sym_set_labels.size() - 1); // the projector's own set
  ASSERT_EQ(Fq.sym_set_sizes(s_int), 1);
  auto Fq_perm = permute_set_to_front(Fq, s_int, ext);

  // check that the permuted set 0 misses the largest block
  int msd = max_subspace_dim(ad);
  ASSERT_LT(nda::max_element(Fq_perm.Fs[0].get_block_sizes()), msd)
     << "vacuous test: symmetry set 0 of the permuted quartet already covers the largest subspace";

  nda::vector<double> hyb_poles = {1.3, -0.8};
  DiagramEvaluator D(beta, Lambda, eps, hyb_poles, ext, Fq_perm, /*n_int=*/1);
  EXPECT_GE(D.Nmax, msd) << "Nmax must be the max block dimension over the WHOLE quartet, not over Fs[0]";
}

/**
 * @brief Check that the symmetry set order is a pure relabeling that does not change the self-energy
 */
TEST(BlockSparseDynintEvaluator, symmetry_set_order_does_not_change_sigma) {

  auto ad       = test_utils::unequal_sym_set_model(true);
  int n_hyb     = static_cast<int>(ad.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
  auto hyb      = test_utils::sym_set_diagonal_hyb(labels_f, p_poles);

  auto Fq = std::get<0>(get_operators(ad, hyb));
  ASSERT_GT(nda::max_element(Fq.sym_set_labels), 0) << "vacuous test: this model has only one symmetry set";
  auto Fq_perm = permute_set_to_front(Fq, 1, hyb); // labels [0,0,1] -> [1,1,0]

  nda::vector<double> hyb_poles = {1.3, -0.8};
  nda::array<int, 2> topology   = {{0, 2}, {1, 3}};
  auto itops                    = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt                       = test_utils::ad_to_atom_prop(ad, beta, itops);

  DiagramEvaluator Da(beta, Lambda, eps, hyb_poles, hyb, Fq);
  DiagramEvaluator Db(beta, Lambda, eps, hyb_poles, hyb, Fq_perm);
  auto Sa = BlockDiagOpFun(Da.compute_self_energy(Gt, topology));
  auto Sb = BlockDiagOpFun(Db.compute_self_energy(Gt, topology));

  double err = 0.0, scale = 0.0;
  for (int b = 0; b < Sa.get_num_block_cols(); ++b) {
    err   = std::max(err, nda::max_element(nda::abs(Sa.get_block(b) - Sb.get_block(b))));
    scale = std::max(scale, nda::max_element(nda::abs(Sa.get_block(b))));
  }
  ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is zero";
  EXPECT_LE(err, 1.0e-14) << "max|Sigma_original - Sigma_permuted| = " << err;
}

/**
 * @brief Compare Sigma and the spgf to the dense evaluator on the interleaved spin-flip model with a dimension-4 subspace
 *
 * @details The other comparisons in this file run on unequal_sym_set_model, with a largest subspace of dimension 2 and contiguous labels.
 * The spin-flip model with autopartitioning has 9 subspaces including one of dimension 4, interleaved symmetry-set labels {0,1,0,1}, and a
 * dynint symmetry set of size 2, since n_up0 and n_do0 share a connection row.
 */
TEST(BlockSparseDynintEvaluator, matches_dense_on_the_interleaved_deep_subspace_model) {
  auto ad      = test_utils::spin_flip_atom_diag_helper(2, false);
  auto ad_flat = test_utils::spin_flip_atom_diag_helper_single_subspace(2);
  test_utils::assert_parallel_atom_diags(ad, ad_flat);

  int n_hyb     = static_cast<int>(ad.get_fops().size());
  auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
  auto hyb      = test_utils::sym_set_diagonal_hyb(labels_f, p_poles);

  // two density operators on the same orbital share a connection row, hence one dynint set of size 2
  std::vector<many_body_operator_complex> ops{n("up", 0), n("do", 0)};
  int n_int = static_cast<int>(ops.size());
  auto d    = nda::zeros<dcomplex>(p_poles, n_int, n_int);
  for (int l = 0; l < p_poles; ++l)
    for (int i = 0; i < n_int; ++i) d(l, i, i) = 0.53 - 0.06 * l + 0.04 * i;

  nda::vector<double> hyb_poles = {1.3, -0.8};
  auto [Fq, labels]             = get_operators_and_interactions(ad, hyb, d, ops);
  auto ext                      = get_extended_coefficients(hyb, d);

  ASSERT_EQ(nda::max_element(Fq.sym_set_sizes), 2) << "every symmetry set is a singleton: the within-set contraction is not exercised";
  ASSERT_EQ(Fq.sym_set_sizes(nda::max_element(Fq.sym_set_labels)), 2)
     << "the dynint operators did not land in one shared set of size 2, so this test duplicates dynint_model. "
        "sym_set_sizes = "
     << Fq.sym_set_sizes;
  {
    auto lab = nda::vector<long>(labels_f);
    std::sort(lab.begin(), lab.end());
    bool interleaved = !std::equal(lab.begin(), lab.end(), labels_f.begin());
    ASSERT_TRUE(interleaved) << "vacuous test: the fermionic labels " << labels_f
                             << " are already sorted, so set-major and "
                                "orbital order coincide and an ordering bug would be invisible";
  }

  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = test_utils::ad_to_atom_prop(ad, beta, itops);
  DiagramEvaluator D(beta, Lambda, eps, hyb_poles, ext, Fq, n_int);

  auto G_flat = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  dense::DiagramEvaluator D_dense(hyb_poles, hyb, G_flat[0].mesh(), ad_flat, ops, d);

  int max_dim = 0;
  for (auto dim : ad.get_subspace_dims()) max_dim = std::max(max_dim, static_cast<int>(dim));
  ASSERT_GE(max_dim, 4) << "vacuous test: this fixture is supposed to be the deep-subspace one, got max dim " << max_dim;
  ASSERT_EQ(D.Nmax, max_dim);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  // Sigma, per block, through the basis bridge.
  {
    SCOPED_TRACE("self-energy");
    auto Sigma        = BlockDiagOpFun(D.compute_self_energy(Gt, topology));
    auto Sigma_dense  = D_dense.compute_self_energy(G_flat, topology);
    auto [err, scale] = test_utils::compare_sigma_with_dense(Sigma, Sigma_dense, ad);
    ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is too small for an absolute tolerance";
    EXPECT_LE(err, sigma_tol * std::max(1.0, scale)) << "max|Sigma_dense| = " << scale << ", max|bs - dense| = " << err;
  }

  // spgf, a trace quantity without bridge
  {
    SCOPED_TRACE("single-particle Green's function");
    auto spgf_dense   = D_dense.compute_single_ptcle_gf(G_flat, topology);
    auto [err, scale] = test_utils::compare_spgf(D.compute_single_ptcle_gf(Gt, topology), spgf_dense);
    ASSERT_GT(scale, 1.0e-3) << "vacuous test: the spgf is too small to compare";
    EXPECT_LE(err, sigma_tol * std::max(1.0, scale)) << "max|spgf_dense| = " << scale << ", max|bs - dense| = " << err;
  }
}

TEST(BlockSparseDynintEvaluator, coupled_density_operators_match_dense_on_one_state_sectors) {
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("x", 0);
  fops.insert("x", 1);
  many_body_operator_complex H = 0.3 * n("x", 0) + 0.7 * n("x", 1) + 0.6 * n("x", 0) * n("x", 1);
  triqs::atom_diag::atom_diag<true> partitioned(H, fops), flat(H, fops, std::vector<many_body_operator_complex>{});
  ASSERT_EQ(partitioned.n_subspaces(), 4);
  ASSERT_EQ(flat.n_subspaces(), 1);
  auto hyb = nda::zeros<dcomplex>(1, 2, 2);
  hyb(0, 0, 0) = 0.4;
  hyb(0, 1, 1) = 0.6;
  auto interaction = nda::zeros<dcomplex>(1, 2, 2);
  interaction(0, 0, 0) = 0.31;
  interaction(0, 1, 1) = 0.47;
  interaction(0, 0, 1) = dcomplex(0.23, 0.07);
  interaction(0, 1, 0) = dcomplex(0.19, -0.11);
  std::vector<many_body_operator_complex> operators{n("x", 0), n("x", 1)};
  nda::vector<double> poles{0.6};
  auto G = test_utils::ad_to_atom_prop(partitioned, beta, Lambda, eps);
  auto dense_G = test_utils::ad_to_atom_prop(flat, beta, Lambda, eps);
  DiagramEvaluator sparse(poles, hyb, G[0].mesh(), partitioned, operators, interaction);
  dense::DiagramEvaluator dense(poles, hyb, dense_G[0].mesh(), flat, operators, interaction);
  for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}},
                         nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}}) {
    SCOPED_TRACE(::testing::Message() << "order=" << topology.extent(0));
    auto sigma = BlockDiagOpFun(sparse.compute_self_energy(G, topology));
    auto reference = dense.compute_self_energy(dense_G, topology);
    auto [error, scale] = test_utils::compare_sigma_with_dense(sigma, reference, partitioned);
    ASSERT_GT(scale, 1e-7);
    EXPECT_LE(error, 2e-10 * scale);
    auto spgf = sparse.compute_single_ptcle_gf(G, topology);
    auto dense_spgf = dense.compute_single_ptcle_gf(dense_G, topology);
    EXPECT_LE(nda::max_element(nda::abs(spgf - dense_spgf)), 2e-10 * nda::max_element(nda::abs(dense_spgf)));
    nda::vector<long> indices(sparse.get_num_single_ptcle_gf_backbones(topology));
    for (long i = 0; i < indices.size(); ++i) indices(i) = i;
    auto correlator = sparse.compute_one_time_correlator(G, operators, operators, partitioned, topology, indices);
    auto dense_correlator = dense.compute_one_time_correlator(dense_G, operators, operators, flat, topology, indices);
    EXPECT_LE(nda::max_element(nda::abs(correlator - dense_correlator)), 2e-10 * nda::max_element(nda::abs(dense_correlator)));
  }
}

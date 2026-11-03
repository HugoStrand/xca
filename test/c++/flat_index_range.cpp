#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"
#include "test_utils/evaluators.hpp"
#include "test_utils/prepared_evaluators.hpp"

using nda::dcomplex;

using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

// Historical fixture ordinals are decoded into prepared Backbones by test-only
// helpers. Large-index numerical checks remain useful even though production
// evaluators no longer expose index selection or count methods.

namespace {

  // Reject the invalid selection with a relevant diagnostic, independent of the internal validation function.
  template <typename F> void expect_flat_index_rejected(F &&call) {
    try {
      call();
      ADD_FAILURE() << "no exception for an out-of-range flat index";
    } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("flat index"), std::string::npos) << e.what(); }
  }

} // namespace

/// Fixture decoding rejects invalid ordinals; failed fixture accumulation does not contaminate a subsequent topology call.
TEST(FlatIndexRange, out_of_range_flat_indices_are_rejected) {
  double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  int p = 2;

  auto ad          = test_utils::unequal_sym_set_model(true);
  auto ad_flat     = test_utils::unequal_sym_set_model(false);
  int n_hyb        = static_cast<int>(ad.get_fops().size());
  auto labels      = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  auto hyb_coeffs  = test_utils::sym_set_diagonal_hyb(labels, p);
  auto hyb_poles   = nda::vector<double>{1.3, -0.8};
  auto G_bs        = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
  auto G_flat      = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  auto ops         = std::vector<many_body_operator_complex>{n("A", 0)};
  auto topology    = nda::array<int, 2>{{0, 2}, {1, 3}};
  auto D_dense     = dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat);
  auto D_bs        = block_sparse::DiagramEvaluator(hyb_poles, hyb_coeffs, G_bs[0].mesh(), ad);
  int n_sigma      = test_utils::fixture_count(D_dense, topology, triqs_xca::diagram::Kind::self_energy);
  int n_correlator = test_utils::fixture_count(D_dense, topology, triqs_xca::diagram::Kind::correlator);
  ASSERT_EQ(test_utils::fixture_count(D_bs, topology, triqs_xca::diagram::Kind::self_energy), n_sigma);
  ASSERT_EQ(test_utils::fixture_count(D_bs, topology, triqs_xca::diagram::Kind::correlator), n_correlator);

  // the first n_valid flat indices, and the same followed by bad
  auto first = [](int n_valid) {
    auto v = nda::vector<long>(n_valid);
    for (int i = 0; i < n_valid; ++i) v(i) = i;
    return v;
  };
  auto valid_then = [&](int n_valid, int bad) {
    auto v                 = nda::vector<long>(n_valid + 1);
    v(nda::range(n_valid)) = first(n_valid);
    v(n_valid)             = bad;
    return v;
  };

  for (int bad : {-1, n_sigma}) {
    SCOPED_TRACE("by-pairs self-energy, f_ix = " + std::to_string(bad));
    expect_flat_index_rejected([&] { test_utils::fixture_self_energy(D_dense, G_flat, topology, bad); });
    expect_flat_index_rejected([&] { test_utils::fixture_self_energy(D_dense, G_flat, topology, valid_then(n_sigma, bad)); });
  }
  for (int bad : {-1, n_correlator}) {
    SCOPED_TRACE("one-time correlators, f_ix = " + std::to_string(bad));
    expect_flat_index_rejected([&] { test_utils::fixture_one_time_correlator(D_dense, G_flat, ops, ops, ad_flat, topology, valid_then(n_correlator, bad)); });
    expect_flat_index_rejected([&] { test_utils::fixture_one_time_correlator(D_bs, G_bs, ops, ops, ad, topology, valid_then(n_correlator, bad)); });
  }

  // A topology call or a new fixture evaluation resets a previous partial sum.
  auto fresh = [&] { return dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat); };
  auto fail  = [&] { EXPECT_THROW(test_utils::fixture_self_energy(D_dense, G_flat, topology, valid_then(n_sigma, n_sigma)), std::invalid_argument); };
  auto check = [&](std::string const &what, auto const &Sigma, auto const &Sigma_fresh) {
    SCOPED_TRACE(what + " after a failed call");
    ASSERT_GT(nda::max_element(nda::abs(Sigma_fresh[0].data())), 0.01);
    EXPECT_EQ(nda::max_element(nda::abs(Sigma[0].data() - Sigma_fresh[0].data())), 0.0);
  };

  fail();
  check("compute_self_energy_by_pairs(G, topology)", D_dense.compute_self_energy_by_pairs(G_flat, topology),
        fresh().compute_self_energy_by_pairs(G_flat, topology));

}

/// Decode fixture ordinals beyond int range, then compare local prepared execution in both evaluators.
TEST(FlatIndexRange, flat_indices_beyond_int) {
  double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  int m = 6, p = 20;

  auto ad         = test_utils::unequal_sym_set_model(true);
  auto ad_flat    = test_utils::unequal_sym_set_model(false);
  int n_hyb       = static_cast<int>(ad.get_fops().size());
  auto labels     = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  auto hyb_coeffs = test_utils::sym_set_diagonal_hyb(labels, p);
  auto hyb_poles  = nda::vector<double>(p);
  for (int l = 0; l < p; ++l) hyb_poles(l) = -2.0 + 0.2 * l;
  auto G_bs     = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
  auto G_flat   = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  auto D_dense  = dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat);
  auto D_bs     = block_sparse::DiagramEvaluator(hyb_poles, hyb_coeffs, G_bs[0].mesh(), ad);
  auto topology = nda::array<int, 2>(m, 2);
  for (int i = 0; i < m; ++i) topology(i, nda::range::all) = nda::vector<int>{i, i + m}; // maximally crossing

  // 2^m n^(m-1) p^(m-1) self-energy and 2^(m-1) n^(m-1) p^(m-1) single-particle Green's function backbones
  long n_sigma = test_utils::fixture_count(D_dense, topology, triqs_xca::diagram::Kind::self_energy);
  long n_spgf  = test_utils::fixture_count(D_dense, topology, triqs_xca::diagram::Kind::correlator);
  ASSERT_EQ(n_sigma, 49766400000L);
  ASSERT_EQ(n_spgf, 24883200000L);
  ASSERT_EQ(test_utils::fixture_count(D_bs, topology, triqs_xca::diagram::Kind::self_energy), n_sigma);
  ASSERT_EQ(test_utils::fixture_count(D_bs, topology, triqs_xca::diagram::Kind::correlator), n_spgf);

  // flat indices with the highest pole indices and orbital index 0 are f = n_op (fb + 1) - n_hyb^(m-1) for the line directions fb,
  // where n_op = n_hyb^(m-1) p^(m-1); the highest fb whose backbone contributes is compared between the evaluators
  long n_op = n_sigma >> m, n_o = 1;
  for (int i = 0; i < m - 1; ++i) n_o *= n_hyb;
  auto sigma_dense = [&](long f) { return nda::array<dcomplex, 3>(test_utils::fixture_self_energy(D_dense, G_flat, topology, f)[0].data()); };
  auto sigma_bs    = [&](long f) { return test_utils::get_tensor_in_full_hilbert_space(test_utils::fixture_self_energy(D_bs, G_bs, topology, f), ad); };
  long f_sigma     = 0;
  for (long fb = (1L << m) - 1; fb >= 0 && f_sigma == 0; --fb)
    if (nda::max_element(nda::abs(sigma_dense(n_op * (fb + 1) - n_o))) > 0.0) f_sigma = n_op * (fb + 1) - n_o;
  ASSERT_GT(f_sigma, long{std::numeric_limits<int>::max()});
  auto S = sigma_dense(f_sigma);
  EXPECT_LE(nda::max_element(nda::abs(sigma_bs(f_sigma) - S)), 1.0e-13 * nda::max_element(nda::abs(S)));

  // Fixture selections and prepared pair execution include the backbone whose line at vertex 0 has the other
  // direction, i.e. the other lowest bit of fb
  auto f_sigma_vec = nda::vector<long>{f_sigma};
  EXPECT_LE(nda::max_element(nda::abs(test_utils::fixture_self_energy(D_dense, G_flat, topology, f_sigma_vec)[0].data() - S)),
            1.0e-13 * nda::max_element(nda::abs(S)));
  EXPECT_LE(nda::max_element(nda::abs(test_utils::get_tensor_in_full_hilbert_space(test_utils::fixture_self_energy(D_bs, G_bs, topology, f_sigma_vec), ad) - S)),
            1.0e-13 * nda::max_element(nda::abs(S)));
  long partner = (f_sigma / n_op) % 2 == 0 ? f_sigma + n_op : f_sigma - n_op;
  auto pair    = nda::make_regular(S + sigma_dense(partner));
  D_dense.reset();
  auto pair_backbone = triqs_xca::diagram::prepare_backbone<triqs_xca::diagram::Kind::self_energy>(topology, D_dense.action, f_sigma);
  D_dense.eval_self_energy_pair(G_flat[0].data(), pair_backbone);
  EXPECT_LE(nda::max_element(nda::abs(D_dense.Sigma - pair)), 1.0e-13 * nda::max_element(nda::abs(pair)));

  long f_spgf = 0;
  for (long fb = (1L << (m - 1)) - 1; fb >= 0 && f_spgf == 0; --fb)
    if (nda::max_element(nda::abs(test_utils::fixture_single_ptcle_gf(D_dense, G_flat, topology, n_op * (fb + 1) - n_o))) > 0.0) f_spgf = n_op * (fb + 1) - n_o;
  ASSERT_GT(f_spgf, long{std::numeric_limits<int>::max()});
  auto g = test_utils::fixture_single_ptcle_gf(D_dense, G_flat, topology, f_spgf);
  EXPECT_LE(nda::max_element(nda::abs(test_utils::fixture_single_ptcle_gf(D_bs, G_bs, topology, f_spgf) - g)), 1.0e-13 * nda::max_element(nda::abs(g)));

  // Fixture selections and the correlator of annihilators and creators agree with the single-particle Green's function.
  auto f_spgf_vec = nda::vector<long>{f_spgf};
  auto c_ops = std::vector<many_body_operator_complex>{}, c_dag_ops = std::vector<many_body_operator_complex>{};
  for (auto const &fop : ad.get_fops()) {
    c_ops.push_back(many_body_operator_complex::make_canonical(false, fop.index));
    c_dag_ops.push_back(many_body_operator_complex::make_canonical(true, fop.index));
  }
  auto check_g = [&](std::string const &what, nda::array_const_view<dcomplex, 3> x) {
    SCOPED_TRACE(what);
    EXPECT_LE(nda::max_element(nda::abs(x - g)), 1.0e-13 * nda::max_element(nda::abs(g)));
  };
  check_g("dense spgf", test_utils::fixture_single_ptcle_gf(D_dense, G_flat, topology, f_spgf_vec));
  check_g("block-sparse spgf", test_utils::fixture_single_ptcle_gf(D_bs, G_bs, topology, f_spgf_vec));
  check_g("dense correlator", test_utils::fixture_one_time_correlator(D_dense, G_flat, c_ops, c_dag_ops, ad_flat, topology, f_spgf_vec));
  check_g("block-sparse correlator", test_utils::fixture_one_time_correlator(D_bs, G_bs, c_ops, c_dag_ops, ad, topology, f_spgf_vec));


  EXPECT_THROW(test_utils::fixture_self_energy(D_dense, G_flat, topology, n_sigma), std::invalid_argument);
  EXPECT_THROW(test_utils::fixture_single_ptcle_gf(D_bs, G_bs, topology, n_spgf), std::invalid_argument);
}

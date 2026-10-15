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

using nda::dcomplex;

using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

/**
 * @file flat_index_range.cpp
 *
 * @brief The range of the flat indices of both diagram evaluators: the routines outside the API that flat_index_all_evals.cpp checks, and
 * flat indices beyond the range of an int
 *
 * @details Both evaluators use shared flat-index decoding, which throws std::invalid_argument outside [0, number of
 * backbones). The flat-index overloads of the self-energy and the single-particle Green's function, and the calls that follow a call
 * that threw, are checked for every evaluator in flat_index_all_evals.cpp; here the dense by-pairs self-energy and the one-time
 * correlators, and every flat-index overload at flat indices beyond the range of an int. The dense self-energy accumulates in a member,
 * so a call that throws part way through a flat-index vector must not leave a partial sum behind for the by-pairs overloads. Assignment
 * counts and flat indices are longs, since the counts exceed the range of an int already at moderate order.
 */

namespace {

  // Reject the invalid selection with a relevant diagnostic, independent of the internal validation function.
  template <typename F> void expect_flat_index_rejected(F &&call) {
    try {
      call();
      ADD_FAILURE() << "no exception for an out-of-range flat index";
    } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("flat index"), std::string::npos) << e.what(); }
  }

} // namespace

/// @brief Check the flat-index overloads of the dense by-pairs self-energy and of the one-time correlators, and the by-pairs overloads
/// after a failed call
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
  int n_sigma      = D_dense.get_num_self_energy_backbones(topology);
  int n_correlator = D_dense.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_EQ(D_bs.get_num_self_energy_backbones(topology), n_sigma);
  ASSERT_EQ(D_bs.get_num_single_ptcle_gf_backbones(topology), n_correlator);

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
  auto all_sigma = first(n_sigma);

  for (int bad : {-1, n_sigma}) {
    SCOPED_TRACE("by-pairs self-energy, f_ix = " + std::to_string(bad));
    expect_flat_index_rejected([&] { D_dense.compute_self_energy_by_pairs(G_flat, topology, bad); });
    expect_flat_index_rejected([&] { D_dense.compute_self_energy_by_pairs(G_flat, topology, valid_then(n_sigma, bad)); });
  }
  for (int bad : {-1, n_correlator}) {
    SCOPED_TRACE("one-time correlators, f_ix = " + std::to_string(bad));
    expect_flat_index_rejected([&] { D_dense.compute_one_time_correlator(G_flat, ops, ops, ad_flat, topology, valid_then(n_correlator, bad)); });
    expect_flat_index_rejected([&] { D_bs.compute_one_time_correlator(G_bs, ops, ops, ad, topology, valid_then(n_correlator, bad)); });
  }

  // each dense by-pairs overload, called after a call that failed with a partial sum in Sigma, agrees with a fresh evaluator
  auto fresh = [&] { return dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat); };
  auto fail  = [&] { EXPECT_THROW(D_dense.compute_self_energy(G_flat, topology, valid_then(n_sigma, n_sigma)), std::invalid_argument); };
  auto check = [&](std::string const &what, auto const &Sigma, auto const &Sigma_fresh) {
    SCOPED_TRACE(what + " after a failed call");
    ASSERT_GT(nda::max_element(nda::abs(Sigma_fresh[0].data())), 0.01);
    EXPECT_EQ(nda::max_element(nda::abs(Sigma[0].data() - Sigma_fresh[0].data())), 0.0);
  };
  int f_nonzero = 0;
  while (nda::max_element(nda::abs(fresh().compute_self_energy(G_flat, topology, f_nonzero)[0].data())) < 0.01) ++f_nonzero;

  fail();
  check("compute_self_energy_by_pairs(G, topology)", D_dense.compute_self_energy_by_pairs(G_flat, topology),
        fresh().compute_self_energy_by_pairs(G_flat, topology));
  fail();
  check("compute_self_energy_by_pairs(G, topology, f_ix)", D_dense.compute_self_energy_by_pairs(G_flat, topology, f_nonzero),
        fresh().compute_self_energy_by_pairs(G_flat, topology, f_nonzero));
  fail();
  check("compute_self_energy_by_pairs(G, topology, f_ix_vec)", D_dense.compute_self_energy_by_pairs(G_flat, topology, all_sigma),
        fresh().compute_self_energy_by_pairs(G_flat, topology, all_sigma));
}

/// @brief Check that both evaluators count and evaluate backbones whose flat index is beyond the range of an int, and agree on them
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
  long n_sigma = D_dense.get_num_self_energy_backbones(topology);
  long n_spgf  = D_dense.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_EQ(n_sigma, 49766400000L);
  ASSERT_EQ(n_spgf, 24883200000L);
  ASSERT_EQ(D_bs.get_num_self_energy_backbones(topology), n_sigma);
  ASSERT_EQ(D_bs.get_num_single_ptcle_gf_backbones(topology), n_spgf);

  // flat indices with the highest pole indices and orbital index 0 are f = n_op (fb + 1) - n_hyb^(m-1) for the line directions fb,
  // where n_op = n_hyb^(m-1) p^(m-1); the highest fb whose backbone contributes is compared between the evaluators
  long n_op = n_sigma >> m, n_o = 1;
  for (int i = 0; i < m - 1; ++i) n_o *= n_hyb;
  auto sigma_dense = [&](long f) { return nda::array<dcomplex, 3>(D_dense.compute_self_energy(G_flat, topology, f)[0].data()); };
  auto sigma_bs    = [&](long f) { return test_utils::get_tensor_in_full_hilbert_space(D_bs.compute_self_energy(G_bs, topology, f), ad); };
  long f_sigma     = 0;
  for (long fb = (1L << m) - 1; fb >= 0 && f_sigma == 0; --fb)
    if (nda::max_element(nda::abs(sigma_dense(n_op * (fb + 1) - n_o))) > 0.0) f_sigma = n_op * (fb + 1) - n_o;
  ASSERT_GT(f_sigma, long{std::numeric_limits<int>::max()});
  auto S = sigma_dense(f_sigma);
  EXPECT_LE(nda::max_element(nda::abs(sigma_bs(f_sigma) - S)), 1.0e-13 * nda::max_element(nda::abs(S)));

  // the flat-index vector overloads, and the dense by-pairs overload, which adds the backbone whose line at vertex 0 has the other
  // direction, i.e. the other lowest bit of fb
  auto f_sigma_vec = nda::vector<long>{f_sigma};
  EXPECT_LE(nda::max_element(nda::abs(D_dense.compute_self_energy(G_flat, topology, f_sigma_vec)[0].data() - S)),
            1.0e-13 * nda::max_element(nda::abs(S)));
  EXPECT_LE(nda::max_element(nda::abs(test_utils::get_tensor_in_full_hilbert_space(D_bs.compute_self_energy(G_bs, topology, f_sigma_vec), ad) - S)),
            1.0e-13 * nda::max_element(nda::abs(S)));
  EXPECT_LE(nda::max_element(nda::abs(D_dense.compute_self_energy(G_flat, topology, f_sigma, f_sigma + 1)[0].data() - S)),
            1.0e-13 * nda::max_element(nda::abs(S)));
  EXPECT_LE(nda::max_element(nda::abs(test_utils::get_tensor_in_full_hilbert_space(D_bs.compute_self_energy(G_bs, topology, f_sigma, f_sigma + 1), ad) - S)),
            1.0e-13 * nda::max_element(nda::abs(S)));
  long partner = (f_sigma / n_op) % 2 == 0 ? f_sigma + n_op : f_sigma - n_op;
  auto pair    = nda::make_regular(S + sigma_dense(partner));
  EXPECT_LE(nda::max_element(nda::abs(D_dense.compute_self_energy_by_pairs(G_flat, topology, f_sigma)[0].data() - pair)),
            1.0e-13 * nda::max_element(nda::abs(pair)));

  long f_spgf = 0;
  for (long fb = (1L << (m - 1)) - 1; fb >= 0 && f_spgf == 0; --fb)
    if (nda::max_element(nda::abs(D_dense.compute_single_ptcle_gf(G_flat, topology, n_op * (fb + 1) - n_o))) > 0.0) f_spgf = n_op * (fb + 1) - n_o;
  ASSERT_GT(f_spgf, long{std::numeric_limits<int>::max()});
  auto g = D_dense.compute_single_ptcle_gf(G_flat, topology, f_spgf);
  EXPECT_LE(nda::max_element(nda::abs(D_bs.compute_single_ptcle_gf(G_bs, topology, f_spgf) - g)), 1.0e-13 * nda::max_element(nda::abs(g)));

  // the vector overloads, and the one-time correlator of the annihilators and creators, which is the single-particle Green's function
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
  check_g("dense spgf", D_dense.compute_single_ptcle_gf(G_flat, topology, f_spgf_vec));
  check_g("block-sparse spgf", D_bs.compute_single_ptcle_gf(G_bs, topology, f_spgf_vec));
  check_g("dense correlator", D_dense.compute_one_time_correlator(G_flat, c_ops, c_dag_ops, ad_flat, topology, f_spgf_vec));
  check_g("block-sparse correlator", D_bs.compute_one_time_correlator(G_bs, c_ops, c_dag_ops, ad, topology, f_spgf_vec));

  check_g("dense spgf range", D_dense.compute_single_ptcle_gf(G_flat, topology, f_spgf, f_spgf + 1));
  check_g("block-sparse spgf range", D_bs.compute_single_ptcle_gf(G_bs, topology, f_spgf, f_spgf + 1));
  check_g("dense correlator range", D_dense.compute_one_time_correlator(G_flat, c_ops, c_dag_ops, ad_flat, topology, f_spgf, f_spgf + 1));
  check_g("block-sparse correlator range", D_bs.compute_one_time_correlator(G_bs, c_ops, c_dag_ops, ad, topology, f_spgf, f_spgf + 1));

  EXPECT_THROW(D_dense.compute_self_energy(G_flat, topology, n_sigma), std::invalid_argument);
  EXPECT_THROW(D_bs.compute_single_ptcle_gf(G_bs, topology, n_spgf), std::invalid_argument);
}

TEST(FlatIndexRange, bounds_stream_without_materializing_the_assignment_space) {
  using triqs_xca::FlatIndexSelection;
  long count = std::numeric_limits<long>::max();
  std::vector<long> seen;
  FlatIndexSelection{count - 2, count, count}.for_each([&](long i) { seen.push_back(i); });
  EXPECT_EQ(seen, (std::vector<long>{count - 2, count - 1}));
  FlatIndexSelection{count, count, count}.for_each([&](long) { FAIL() << "nonempty terminal range"; });
  FlatIndexSelection{0, 0, 0}.for_each([&](long) { FAIL() << "nonempty zero-count range"; });
  EXPECT_THROW((FlatIndexSelection{-1, 0, count}), std::invalid_argument);
  EXPECT_THROW((FlatIndexSelection{2, 1, count}), std::invalid_argument);
  EXPECT_THROW((FlatIndexSelection{0, 2, 1}), std::invalid_argument);
}

TEST(FlatIndexRange, partitions_preserve_self_energy_pairs_and_observables) {
  auto model = test_utils::build_eval_model("unequal_sets_nA0");
  dense::DiagramEvaluator dense_eval(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  block_sparse::DiagramEvaluator sparse_eval(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
  auto dense_G = test_utils::subspace_blocks(model.G, model.ad_flat, model.mesh);
  auto sparse_G = test_utils::subspace_blocks(model.G, model.ad, model.mesh);
  auto ops = model.dynint_ops; // bosonic observable in a model with both fermionic and retarded interactions
  auto all_indices = [](long count) {
    nda::vector<long> indices(count);
    for (long i = 0; i < count; ++i) indices(i) = i;
    return indices;
  };
  auto check = [](long count, auto const &reference, auto &&evaluate) {
    double scale = nda::max_element(nda::abs(reference));
    ASSERT_GT(scale, 1.e-10);
    auto sum = nda::zeros<dcomplex>(reference.shape());
    // Splits need not align with a direction digit; include ranks with no assignments.
    std::vector<long> bounds{0, 0, 1, count / 3, count - 1, count, count};
    for (int part = 1; part < bounds.size(); ++part) {
      auto result = evaluate(bounds[part - 1], bounds[part]);
      if (bounds[part - 1] == bounds[part]) EXPECT_EQ(nda::max_element(nda::abs(result)), 0.0);
      sum += result;
    }
    EXPECT_LT(nda::max_element(nda::abs(sum - reference)), 1.e-12 * scale);
    EXPECT_THROW(evaluate(-1, 0), std::invalid_argument);
    EXPECT_THROW(evaluate(2, 1), std::invalid_argument);
    EXPECT_THROW(evaluate(0, count + 1), std::invalid_argument);
    EXPECT_THROW(evaluate(count + 1, count + 1), std::invalid_argument);
    // A rejected request must not contaminate a subsequent valid evaluation.
    EXPECT_LT(nda::max_element(nda::abs(evaluate(0, count) - reference)), 1.e-12 * scale);
  };
  for (auto const &topology : std::vector<nda::array<int, 2>>{{{0, 2}, {1, 3}}, {{0, 2}, {1, 4}, {3, 5}}}) {
    auto test_evaluator = [&](auto &evaluator, auto &G, auto const &ad) {
      long n_sigma = evaluator.get_num_self_energy_backbones(topology);
      long n_corr = evaluator.get_num_single_ptcle_gf_backbones(topology);
      auto sigma = test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(G, topology, all_indices(n_sigma)), ad);
      check(n_sigma, sigma, [&](long first, long last) {
        return test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(G, topology, first, last), ad);
      });
      auto spgf = evaluator.compute_single_ptcle_gf(G, topology, all_indices(n_corr));
      check(n_corr, spgf, [&](long first, long last) { return evaluator.compute_single_ptcle_gf(G, topology, first, last); });
      auto corr = evaluator.compute_one_time_correlator(G, ops, ops, ad, topology, all_indices(n_corr));
      check(n_corr, corr, [&](long first, long last) { return evaluator.compute_one_time_correlator(G, ops, ops, ad, topology, first, last); });
      return sigma;
    };
    auto sigma = test_evaluator(dense_eval, dense_G, model.ad_flat);
    test_evaluator(sparse_eval, sparse_G, model.ad);
    check(dense_eval.get_num_self_energy_backbones(topology), sigma, [&](long first, long last) {
      return test_utils::get_tensor_in_full_hilbert_space(dense_eval.compute_self_energy_by_pairs(dense_G, topology, first, last), model.ad_flat);
    });
  }
  // At first order rank 1 owns only a noncanonical seed and contributes zero.
  nda::array<int, 2> first_order{{0, 1}};
  auto empty_pair = dense_eval.compute_self_energy_by_pairs(dense_G, first_order, 1, 2);
  EXPECT_EQ(nda::max_element(nda::abs(empty_pair[0].data())), 0.0);
  auto pair = dense_eval.compute_self_energy_by_pairs(dense_G, first_order, 0, 1);
  auto full = dense_eval.compute_self_energy(dense_G, first_order);
  EXPECT_LT(nda::max_element(nda::abs(pair[0].data() - full[0].data())), 1.e-12);
}

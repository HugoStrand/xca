#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include <triqs_xca/dense/cuda_small/diagram_evaluator.hpp>
#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"

namespace {
  namespace utils = triqs_xca::test_utils;
  using nda::dcomplex;

  template <typename D, typename G, typename AD> void check_ranges(D &d, G &g, AD const &ad) {
    nda::array<int, 2> topology{{0, 2}, {1, 3}};
    long count  = d.get_num_self_energy_backbones(topology);
    auto full   = d.compute_self_energy(g, topology);
    auto ranged = d.compute_self_energy_range(g, topology, 0, count);
    auto pieces = d.compute_self_energy_range(g, topology, 0, count / 3);
    pieces += d.compute_self_energy_range(g, topology, count / 3, count);
    auto empty = d.compute_self_energy_range(g, topology, count, count);
    for (long b = 0; b < full.size(); ++b) {
      EXPECT_LE(nda::max_element(nda::abs(full[b].data() - ranged[b].data())), 1e-12);
      EXPECT_LE(nda::max_element(nda::abs(full[b].data() - pieces[b].data())), 1e-12);
      EXPECT_EQ(nda::max_element(nda::abs(empty[b].data())), 0.);
    }
    for (auto [first, last] : {std::pair{-1L, 1L}, {2L, 1L}, {0L, count + 1}, {count + 1, count + 1}})
      EXPECT_THROW(d.compute_self_energy_range(g, topology, first, last), std::out_of_range);
    auto recovered = d.compute_self_energy_range(g, topology, 0, count);
    for (long b = 0; b < full.size(); ++b) EXPECT_LE(nda::max_element(nda::abs(full[b].data() - recovered[b].data())), 1e-12);

    long nc = d.get_num_single_ptcle_gf_backbones(topology);
    auto gf = d.compute_single_ptcle_gf(g, topology);
    auto gr = d.compute_single_ptcle_gf_range(g, topology, 0, nc);
    auto gp = d.compute_single_ptcle_gf_range(g, topology, 0, nc / 3);
    gp += d.compute_single_ptcle_gf_range(g, topology, nc / 3, nc);
    EXPECT_LE(nda::max_element(nda::abs(gf - gr)), 1e-12);
    EXPECT_LE(nda::max_element(nda::abs(gf - gp)), 1e-12);
    EXPECT_EQ(nda::max_element(nda::abs(d.compute_single_ptcle_gf_range(g, topology, nc, nc))), 0.);
    EXPECT_THROW(d.compute_single_ptcle_gf_range(g, topology, 0, nc + 1), std::out_of_range);

    using triqs::operators::many_body_operator_complex;
    std::vector<many_body_operator_complex> annihilators, creators;
    for (auto const &fop : ad.get_fops()) {
      annihilators.push_back(many_body_operator_complex::make_canonical(false, fop.index));
      creators.push_back(many_body_operator_complex::make_canonical(true, fop.index));
    }
    auto corr = d.compute_one_time_correlator_range(g, annihilators, creators, ad, topology, 0, nc);
    EXPECT_LE(nda::max_element(nda::abs(corr - gf)), 1e-12);
    auto cp = d.compute_one_time_correlator_range(g, annihilators, creators, ad, topology, 0, nc / 3);
    cp += d.compute_one_time_correlator_range(g, annihilators, creators, ad, topology, nc / 3, nc);
    EXPECT_LE(nda::max_element(nda::abs(corr - cp)), 1e-12);
    EXPECT_EQ(nda::max_element(nda::abs(d.compute_one_time_correlator_range(g, annihilators, creators, ad, topology, nc, nc))), 0.);
    EXPECT_THROW(d.compute_one_time_correlator_range(g, annihilators, creators, ad, topology, -1, 0), std::out_of_range);

    if constexpr (requires { d.compute_self_energy_by_pairs_range(g, topology, 0L, count); }) {
      auto pair_full  = d.compute_self_energy_by_pairs(g, topology);
      auto pair_range = d.compute_self_energy_by_pairs_range(g, topology, 0, count);
      for (long b = 0; b < full.size(); ++b) EXPECT_LE(nda::max_element(nda::abs(pair_full[b].data() - pair_range[b].data())), 1e-12);
      // Cross the forward/backward seed boundary with an unaligned range.
      long group    = count / 4;
      auto selected = d.compute_self_energy_by_pairs_range(g, topology, 1, group + 1);
      auto expected = d.compute_self_energy_by_pairs(g, topology, 1);
      for (long i = 2; i < group; ++i) expected += d.compute_self_energy_by_pairs(g, topology, i);
      for (long b = 0; b < full.size(); ++b) EXPECT_LE(nda::max_element(nda::abs(selected[b].data() - expected[b].data())), 1e-12);
      auto skipped = d.compute_self_energy_by_pairs_range(g, topology, group, 2 * group);
      for (long b = 0; b < full.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(skipped[b].data())), 0.);
      EXPECT_THROW(d.compute_self_energy_by_pairs_range(g, topology, 0, count + 1), std::out_of_range);
    }
  }
} // namespace

TEST(EvaluatorRanges, CPUAndCudaMatchFullAndPartitionedEvaluation) {
  auto ad = utils::unequal_sym_set_model(true), flat = utils::unequal_sym_set_model(false);
  int n = ad.get_fops().size(), p = 2;
  auto labels = std::get<1>(triqs_xca::block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, n, n)));
  auto coeffs = utils::sym_set_diagonal_hyb(labels, p);
  nda::vector<double> poles{1.3, -.8};
  auto g  = utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  auto gf = utils::ad_to_atom_prop(flat, 2., 40., 1e-10);
  triqs_xca::dense::DiagramEvaluator dense(poles, coeffs, gf[0].mesh(), flat);
  triqs_xca::block_sparse::DiagramEvaluator sparse(poles, coeffs, g[0].mesh(), ad);
  check_ranges(dense, gf, flat);
  check_ranges(sparse, g, ad);
#ifdef NDA_HAVE_CUDA
  triqs_xca::dense::cuda::DiagramEvaluator cuda(poles, coeffs, gf[0].mesh(), flat);
  check_ranges(cuda, gf, flat);
#endif
#ifdef XCA_HAVE_CUDA_SMALL
  triqs_xca::dense::cuda_small::DiagramEvaluator small(poles, coeffs, gf[0].mesh(), flat);
  check_ranges(small, gf, flat);
#endif
}

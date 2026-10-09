#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>

#include "test_utils/block_sparse.hpp"

using nda::dcomplex;

using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

using mesh_t = triqs::mesh::dlr_imtime;
using bgf_t  = triqs::gfs::block_gf<mesh_t>;

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

/**
 * @file propagator_layout.cpp
 *
 * @brief Both diagram evaluators reject a pseudo-particle propagator that is not in their layout
 *
 * @details The dense evaluator takes one (r, N, N) block over the full Hilbert space, the block-sparse evaluator one (r, d, d) block per
 * invariant subspace of its atom_diag. The layout of the other evaluator, an empty block_gf, a block of the wrong dimension and a propagator
 * on a DLR mesh of another rank are rejected with std::invalid_argument by every block_gf entry point, the BlockDiagOpFun overloads of the
 * block-sparse evaluator and block_sparse::expectation_value. The one-time correlator also rejects an atom_diag other than the one the
 * evaluator was built on, and expectation_value gets no contribution from the blocks of an operator that leave their subspace.
 */

/// @brief Check every wrong layout against every block_gf entry point of both evaluators
TEST(PropagatorLayout, wrong_layouts_are_rejected) {
  double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  int p = 2;

  auto ad         = test_utils::unequal_sym_set_model(true);
  auto ad_flat    = test_utils::unequal_sym_set_model(false);
  int n_hyb       = static_cast<int>(ad.get_fops().size());
  auto labels     = std::get<1>(block_sparse::atom_diag::get_operators(ad, nda::zeros<dcomplex>(p, n_hyb, n_hyb)));
  auto hyb_coeffs = test_utils::sym_set_diagonal_hyb(labels, p);
  auto hyb_poles  = nda::vector<double>{1.3, -0.8};
  auto G_bs       = test_utils::ad_to_atom_prop(ad, beta, Lambda, eps);
  auto G_flat     = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, eps);
  auto ops        = std::vector<many_body_operator_complex>{n("A", 0)};
  auto topology   = nda::array<int, 2>{{0, 1}};
  auto f_ix_vec   = nda::vector<long>{0};
  auto D_dense    = dense::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat);
#ifdef NDA_HAVE_CUDA
  auto D_cuda = dense::cuda::DiagramEvaluator(hyb_poles, hyb_coeffs, G_flat[0].mesh(), ad_flat);
#endif
  auto D_bs       = block_sparse::DiagramEvaluator(hyb_poles, hyb_coeffs, G_bs[0].mesh(), ad);

  // the evaluators accept their own layout
  ASSERT_GT(ad.n_subspaces(), 1);
  ASSERT_NO_THROW(D_dense.compute_self_energy(G_flat, topology));
  ASSERT_NO_THROW(D_bs.compute_self_energy(G_bs, topology));

  // the block-sparse propagator with one block of dimension > 1 replaced by a nonzero 1x1 block
  auto blocks = std::vector<triqs::gfs::gf<mesh_t>>{};
  for (int b = 0; b < G_bs.size(); ++b) blocks.emplace_back(G_bs[b]);
  int b_big = 0;
  while (b_big < G_bs.size() && G_bs[b_big].target_shape()[0] < 2) ++b_big;
  ASSERT_LT(b_big, G_bs.size()) << "vacuous test: no subspace of dimension > 1";
  blocks[b_big]       = triqs::gfs::gf<mesh_t>(G_bs[0].mesh(), nda::ones<dcomplex>(G_bs[0].mesh().size(), 1, 1));
  auto G_bs_wrong_dim = bgf_t{blocks};

  auto G_empty     = bgf_t{std::vector<triqs::gfs::gf<mesh_t>>{}};
  auto G_flat_rank = test_utils::ad_to_atom_prop(ad_flat, beta, Lambda, 1.0e-6);
  auto G_bs_rank   = test_utils::ad_to_atom_prop(ad, beta, Lambda, 1.0e-6);
  ASSERT_NE(G_flat_rank[0].mesh().size(), G_flat[0].mesh().size());

  auto wrong_for_dense = std::vector<std::pair<std::string, bgf_t>>{{"block-sparse layout", G_bs}, {"empty", G_empty}, {"other rank", G_flat_rank}};
  for (auto &[what, G] : wrong_for_dense) {
    SCOPED_TRACE("dense, " + what);
    EXPECT_THROW(D_dense.compute_self_energy(G, topology), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_self_energy(G, topology), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_self_energy(G, topology, 0), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_self_energy(G, topology, 0), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_self_energy(G, topology, f_ix_vec), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_self_energy(G, topology, f_ix_vec), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_self_energy_by_pairs(G, topology), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_self_energy_by_pairs(G, topology), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_self_energy_by_pairs(G, topology, 0), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_self_energy_by_pairs(G, topology, 0), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_self_energy_by_pairs(G, topology, f_ix_vec), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_self_energy_by_pairs(G, topology, f_ix_vec), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_single_ptcle_gf(G, topology), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_single_ptcle_gf(G, topology), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_single_ptcle_gf(G, topology, 0), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_single_ptcle_gf(G, topology, 0), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_single_ptcle_gf(G, topology, f_ix_vec), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_single_ptcle_gf(G, topology, f_ix_vec), std::invalid_argument);
#endif
    EXPECT_THROW(D_dense.compute_one_time_correlator(G, ops, ops, ad_flat, topology, f_ix_vec), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
    EXPECT_THROW(D_cuda.compute_one_time_correlator(G, ops, ops, ad_flat, topology, f_ix_vec), std::invalid_argument);
#endif
  }
  auto wrong_for_bs = std::vector<std::pair<std::string, bgf_t>>{
     {"dense layout", G_flat}, {"empty", G_empty}, {"wrong block dimension", G_bs_wrong_dim}, {"other rank", G_bs_rank}};
  for (auto &[what, G] : wrong_for_bs) {
    SCOPED_TRACE("block-sparse, " + what);
    EXPECT_THROW(D_bs.compute_self_energy(G, topology), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_self_energy(G, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_self_energy(G, topology, f_ix_vec), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(G, topology), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(G, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(G, topology, f_ix_vec), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_one_time_correlator(G, ops, ops, ad, topology, f_ix_vec), std::invalid_argument);
  }

  // the one-time correlator with an atom_diag of another Hilbert space (dense) or other subspaces (block-sparse)
  auto ad_other = test_utils::sz_resolved_atom_diag_helper(1, false);
  ASSERT_NE(ad_other.get_full_hilbert_space_dim(), ad_flat.get_full_hilbert_space_dim());
  EXPECT_THROW(D_dense.compute_one_time_correlator(G_flat, ops, ops, ad_other, topology, f_ix_vec), std::invalid_argument);
#ifdef NDA_HAVE_CUDA
  EXPECT_THROW(D_cuda.compute_one_time_correlator(G_flat,ops,ops,ad_other,topology,f_ix_vec),std::invalid_argument);
#endif
  EXPECT_THROW(D_bs.compute_one_time_correlator(G_bs, ops, ops, ad_flat, topology, f_ix_vec), std::invalid_argument);

  // the BlockDiagOpFun overloads of the block-sparse evaluator, which a block-sparse layout from another model reaches unconverted
  for (auto &[what, G] : wrong_for_bs) {
    if (G.size() == 0) continue; // BlockDiagOpFun cannot hold an empty propagator
    SCOPED_TRACE("block-sparse BlockDiagOpFun, " + what);
    auto Gt = block_sparse::BlockDiagOpFun(G);
    EXPECT_THROW(D_bs.compute_self_energy(Gt, topology), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_self_energy(Gt, topology, 0), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(Gt, topology), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(Gt, topology, 0), std::invalid_argument);
  }
  {
    SCOPED_TRACE("block-sparse BlockDiagOpFun, zero blocks of dimension 1");
    auto Gt = block_sparse::BlockDiagOpFun(G_bs[0].mesh().size(), nda::ones<int>(G_bs.size()));
    EXPECT_THROW(D_bs.compute_self_energy(Gt, topology), std::invalid_argument);
    EXPECT_THROW(D_bs.compute_single_ptcle_gf(Gt, topology), std::invalid_argument);
  }

  // expectation_value with the layouts of its atom_diag and of another one, and with an operator that leaves the subspaces
  auto op = many_body_operator_complex(n("A", 0));
  EXPECT_NO_THROW(block_sparse::expectation_value(op, ad, G_bs));
  for (auto &[what, G] : wrong_for_bs) {
    if (what == "other rank") continue; // expectation_value reads G on its own mesh
    SCOPED_TRACE("expectation_value, " + what);
    EXPECT_THROW(block_sparse::expectation_value(op, ad, G), std::invalid_argument);
  }
  EXPECT_EQ(std::abs(block_sparse::expectation_value(many_body_operator_complex(c_dag("A", 0)), ad, G_bs)), 0.0);
}

#ifdef NDA_HAVE_CUDA
TEST(PropagatorLayout, cuda_strided_input_and_constructor_interfaces) {
  double beta=2.,Lambda=40.,eps=1e-10;
  triqs::atom_diag::fundamental_operator_set fops;
  fops.insert("0",0);
  triqs::operators::many_body_operator_real H=-.3*n<double>("0",0);
  triqs::atom_diag::atom_diag<false> real_ad(H,fops,std::vector<triqs::operators::many_body_operator_real>{});
  triqs::atom_diag::atom_diag<true> complex_ad(H,fops,std::vector<many_body_operator_complex>{});
  auto g=test_utils::ad_to_atom_prop(complex_ad,beta,Lambda,eps);
  nda::vector<double> poles{-1.2,.8};
  nda::array<dcomplex,3> coefficients={{{dcomplex{.7,.1}}},{{dcomplex{.3,-.2}}}};
  dense::DiagramEvaluator cpu(poles,coefficients,g[0].mesh(),real_ad);
  dense::cuda::DiagramEvaluator gpu(poles,coefficients,g[0].mesh(),real_ad);
  auto itops=g[0].mesh().dlr_it();
  auto dimensionless_poles=nda::make_regular(beta*poles);
  dense::DiagramEvaluator cpu_low(beta,eps,itops,dimensionless_poles,coefficients,cpu.Fset);
  dense::cuda::DiagramEvaluator gpu_low(beta,eps,itops,dimensionless_poles,coefficients,cpu.Fset);
  int r=g[0].data().extent(0),N=g[0].data().extent(1);
  nda::array<dcomplex,3> storage(r,N,2*N);
  auto strided=storage(nda::range::all,nda::range::all,nda::range(0,2*N,2));
  strided=g[0].data();
  triqs::gfs::gf_view<mesh_t> view(g[0].mesh(),strided);
  triqs::gfs::block_gf_view<mesh_t> blocks({"full"},{view});
  ASSERT_EQ(blocks[0].data().strides()[2],2);
  nda::array<int,2> topology={{0,2},{1,3}};
  nda::vector<long> subset{0,1,0};
  auto check=[](auto const &a,auto const &b){EXPECT_LE(nda::max_element(nda::abs(a-b)),1e-12);};
  check(cpu.compute_self_energy(blocks,topology)[0].data(),gpu.compute_self_energy(blocks,topology)[0].data());
  check(cpu_low.compute_self_energy(blocks,topology)[0].data(),gpu_low.compute_self_energy(blocks,topology)[0].data());
  check(cpu.compute_single_ptcle_gf(blocks,topology,subset),gpu.compute_single_ptcle_gf(blocks,topology,subset));
  std::vector<many_body_operator_complex> mu{n<double>("0",0)},kap{n<double>("0",0)};
  check(cpu.compute_one_time_correlator(blocks,mu,kap,real_ad,topology,subset),gpu.compute_one_time_correlator(blocks,mu,kap,real_ad,topology,subset));
  mu.push_back(2.*n<double>("0",0));
  check(cpu.compute_one_time_correlator(blocks,mu,kap,real_ad,topology,subset),gpu.compute_one_time_correlator(blocks,mu,kap,real_ad,topology,subset));
  mu[1]=3.*n<double>("0",0);
  strided*=dcomplex{.9,.03};
  check(cpu.compute_one_time_correlator(blocks,mu,kap,real_ad,topology,subset),gpu.compute_one_time_correlator(blocks,mu,kap,real_ad,topology,subset));
  std::vector<many_body_operator_complex> empty_ops;
  auto empty=gpu.compute_one_time_correlator(blocks,empty_ops,kap,real_ad,topology,subset);
  EXPECT_EQ(empty.shape(),(std::array<long,3>{r,0,1}));
  nda::vector<long> bad{gpu.get_num_single_ptcle_gf_backbones(topology)};
  EXPECT_THROW(gpu.compute_one_time_correlator(blocks,empty_ops,kap,real_ad,topology,bad),std::invalid_argument);
  std::vector<many_body_operator_complex> zero{many_body_operator_complex{}};
  EXPECT_THROW(gpu.compute_one_time_correlator(blocks,zero,kap,real_ad,topology,subset),std::runtime_error);
}
#endif

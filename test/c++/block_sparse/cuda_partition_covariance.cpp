#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include "test_utils/block_sparse.hpp"

namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using nda::dcomplex;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;

  auto partitioned_atom(int partition) {
    triqs::atom_diag::fundamental_operator_set fops;
    many_body_operator_complex h, q;
    for (int i = 0; i < 3; ++i) {
      fops.insert("f", i);
      q += n("f", i);
      h += (.13 + .07 * i) * n("f", i);
      for (int j = 0; j < i; ++j) h += .2 * n("f", i) * n("f", j);
    }
    if (partition == 0) return triqs::atom_diag::atom_diag<true>(h, fops);
    return triqs::atom_diag::atom_diag<true>(h, fops, partition == 1 ? std::vector<many_body_operator_complex>{q}
                                                                 : std::vector<many_body_operator_complex>{});
  }

  auto full_fock(auto const &output, auto const &ad) {
    auto result = nda::zeros<dcomplex>(output[0].data().extent(0), 8, 8);
    for (int b = 0; b < ad.n_subspaces(); ++b) {
      auto const &states = ad.get_fock_states(b);
      for (long i = 0; i < long(states.size()); ++i)
        for (long j = 0; j < long(states.size()); ++j)
          for (long t = 0; t < result.extent(0); ++t) result(t, states[i], states[j]) = output[b].data()(t, i, j);
    }
    return result;
  }

  void compare(auto const &actual, auto const &expected) {
    ASSERT_EQ(actual.shape(), expected.shape());
    double norm = nda::max_element(nda::abs(expected));
    ASSERT_GT(norm, 1e-20);
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 1e-13 * norm);
  }
}

TEST(BlockSparseCudaPartition, SameModelAgainstDenseCpu) {
  // Identical H, hybridization and retarded operators in every partition. Never
  // discard coefficients to force compatibility with the fine partition.
  auto coefficients = nda::zeros<dcomplex>(2, 3, 3);
  for (int p = 0; p < 2; ++p)
    for (int i = 0; i < 3; ++i) coefficients(p, i, i) = .3 - .1 * p + .017 * i;
  nda::vector<double> poles{1.3, -.8};
  std::vector<many_body_operator_complex> ops{n("f", 0) + n("f", 1) + n("f", 2), n("f", 0)};
  auto interactions = nda::zeros<dcomplex>(2, 2, 2);
  interactions(0, 0, 0) = .07;
  interactions(1, 1, 1) = {.03, .01};
  auto dense_atom = partitioned_atom(2);
  ASSERT_EQ(dense_atom.n_subspaces(), 1);
  for (int i = 0; i < 8; ++i) ASSERT_EQ(dense_atom.get_fock_states(0)[i], i);
  auto dense_g = triqs_xca::test_utils::ad_to_atom_prop(dense_atom, 2., 40., 1e-10);
  for (bool dynamic : {false, true}) {
    SCOPED_TRACE(dynamic);
    auto dense = dynamic ? std::make_unique<triqs_xca::dense::DiagramEvaluator>(poles, coefficients, dense_g[0].mesh(), dense_atom, ops, interactions)
                         : std::make_unique<triqs_xca::dense::DiagramEvaluator>(poles, coefficients, dense_g[0].mesh(), dense_atom);
    for (auto topology : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 2}, {1, 3}}}) {
      auto expected_sigma = full_fock(dense->compute_self_energy(dense_g, topology), dense_atom);
      auto expected_spgf = dense->compute_single_ptcle_gf(dense_g, topology);
      auto expected_corr = dense->compute_one_time_correlator_range(dense_g, ops, ops, dense_atom, topology, 0,
                                                                  dense->get_num_single_ptcle_gf_backbones(topology));
      for (int partition : {0, 1, 2}) {
        SCOPED_TRACE(partition);
        auto ad = partitioned_atom(partition);
        EXPECT_EQ(ad.n_subspaces(), partition == 0 ? 8 : partition == 1 ? 4 : 1);
        auto g = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
        compare(full_fock(g, ad), full_fock(dense_g, dense_atom));
        auto model = dynamic ? gpu::export_model(ad, coefficients, ops, interactions) : gpu::export_model(ad, coefficients);
        for (bool native : {false, true}) {
          if (native && (!gpu::ordered_sparse_available() || !gpu::mapped_special_available())) continue;
          SCOPED_TRACE(native);
          gpu::DiagramEvaluator device(model, poles, g[0].mesh(), 1024ULL*1024*1024, 32ULL*1024*1024,
                                       native ? gpu::SparseBackend::Ordered : gpu::SparseBackend::Library,
                                       native ? gpu::SpecialBackend::Mapped : gpu::SpecialBackend::Reference);
          compare(full_fock(device.compute_self_energy(g, topology), ad), expected_sigma);
          compare(device.compute_single_ptcle_gf(g, topology), expected_spgf);
          compare(device.compute_one_time_correlator_range(g, ops, ops, ad, topology, 0,
                                                          device.get_num_single_ptcle_gf_backbones(topology)), expected_corr);
        }
      }
    }
  }
}
#else
TEST(BlockSparseCudaPartition, DISABLED_RequiresCuda) {}
#endif

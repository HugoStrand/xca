#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/path.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include "test_utils/block_sparse.hpp"

namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  namespace bs  = triqs_xca::block_sparse;
  using nda::dcomplex;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  auto atom(int modes, bool number) {
    triqs::atom_diag::fundamental_operator_set fops;
    many_body_operator_complex H, Q;
    for (int i = 0; i < modes; ++i) {
      fops.insert("f", i);
      H += (.11 + .12 * i) * n("f", i);
      Q += n("f", i);
    }
    if (number) return triqs::atom_diag::atom_diag<true>(H, fops, std::vector<many_body_operator_complex>{Q});
    return triqs::atom_diag::atom_diag<true>(H, fops);
  }
  auto coefficients(int modes) {
    auto out = nda::zeros<dcomplex>(2, modes, modes);
    for (int i = 0; i < modes; ++i) {
      out(0, i, i) = .3 + .1 * i;
      out(1, i, i) = .17 - .05 * i;
    }
    return out;
  }
  std::vector<gpu::SparseOperator const *> fields(gpu::OperatorModel const &model, bool dag) {
    std::vector<gpu::SparseOperator const *> result;
    for (int i = 0; i < model.num_hybridization_fields(); ++i) result.push_back(&model.field(i, dag));
    return result;
  }
  int block(triqs::atom_diag::atom_diag<true> const &ad, int state) {
    for (int b = 0; b < ad.n_subspaces(); ++b)
      if (ad.get_fock_states(b).size() == 1 && ad.get_fock_states(b)[0] == state) return b;
    throw std::runtime_error("Missing scalar Fock state");
  }
} // namespace
TEST(CorrelatorPath, EmptySidesAndExactScalarEndpoints) {
  auto ad    = atom(2, false);
  auto model = gpu::export_model(ad, coefficients(2));
  nda::array<int, 2> topology{{0, 1}};
  nda::vector<double> poles{1.3, -.8};
  CorrelatorBackbone backbone(topology, 2);
  backbone.set_flat_index(0, poles);
  std::vector<bool> live(4, true);
  auto mu = fields(model, false), kap = fields(model, true);
  auto fragments = gpu::collect_correlator_fragments(model, backbone, live);
  for (int b = 0; b < 4; ++b)
    for (auto const *side : {&fragments.right, &fragments.left}) {
      ASSERT_TRUE((*side)[b]);
      EXPECT_EQ((*side)[b]->propagator_blocks, std::vector<int>{b});
      EXPECT_EQ((*side)[b]->dimensions, std::vector<long>{1});
    }
  std::size_t visits = 0;
  auto counts =
     gpu::for_each_correlator_endpoint(model, fragments, live, mu, kap, [&](gpu::CorrelatorEndpoint const &e, auto const &right, auto const &left) {
       EXPECT_EQ(e.mu, e.kap);
       EXPECT_EQ(right.root_block, e.right_root);
       EXPECT_EQ(left.root_block, e.left_root);
       auto state = ad.get_fock_states(e.right_root)[0];
       EXPECT_NE(state & (1 << e.mu), 0);
       EXPECT_EQ(ad.get_fock_states(e.left_root)[0], state ^ (1 << e.mu));
       ++visits;
     });
  EXPECT_EQ(visits, 4);
  EXPECT_EQ(counts.proposed, 16);
  EXPECT_EQ(counts.accepted, 4);
  EXPECT_EQ(counts.pruned, 12);
  live.assign(4, false);
  EXPECT_THROW(gpu::for_each_correlator_endpoint(model, fragments, live, mu, kap, [](auto const &, auto const &, auto const &) {}),
               std::invalid_argument);
  auto empty = gpu::collect_correlator_fragments(model, backbone, live);
  for (auto const &p : empty.right) EXPECT_TRUE(p); // Empty walks still retain roots.
  counts =
     gpu::for_each_correlator_endpoint(model, empty, live, mu, kap, [](auto const &, auto const &, auto const &) { FAIL() << "Masked endpoint"; });
  EXPECT_EQ(counts.accepted, 0);
  EXPECT_EQ(counts.pruned, 16);
  EXPECT_THROW(gpu::collect_correlator_fragments(model, backbone, live, 7), std::length_error);
  auto stale                    = empty;
  stale.right[0]->dimensions[0] = 2;
  EXPECT_THROW(gpu::for_each_correlator_endpoint(model, stale, live, mu, kap, [](auto const &, auto const &, auto const &) {}),
               std::invalid_argument);
  std::vector<gpu::SparseOperator const *> bad{nullptr};
  EXPECT_THROW(gpu::for_each_correlator_endpoint(model, empty, live, bad, kap, [](auto const &, auto const &, auto const &) {}),
               std::invalid_argument);
}
TEST(CorrelatorPath, MaskedStartingRootStillContributesToCpuCorrelator) {
  auto ad    = atom(2, false);
  auto coeff = coefficients(2);
  auto model = gpu::export_model(ad, coeff);
  nda::vector<double> poles{1.3, -.8};
  nda::array<int, 2> topology{{0, 2}, {1, 3}};
  auto mu = fields(model, false), kap = fields(model, true);
  CorrelatorBackbone backbone(topology, 2);
  long index = -1;
  for (long i = 0; i < backbone.num_flat_indices(poles.size()); ++i) {
    backbone.set_flat_index(i, poles);
    if (backbone.get_vertex_orb(1) == 0 && !backbone.has_vertex_dag(1)) {
      index = i;
      break;
    }
  }
  ASSERT_GE(index, 0);
  int root = block(ad, 3);
  std::vector<bool> live(4, true);
  live[root]     = false;
  auto fragments = gpu::collect_correlator_fragments(model, backbone, live);
  ASSERT_TRUE(fragments.right[root]);
  EXPECT_EQ(fragments.right[root]->end_block, block(ad, 2));
  bool found = false;
  gpu::for_each_correlator_endpoint(model, fragments, live, mu, kap, [&](gpu::CorrelatorEndpoint const &e, auto const &, auto const &) {
    if (e.mu == 1 && e.kap == 1 && e.right_root == root) {
      found = true;
      EXPECT_EQ(e.left_root, block(ad, 0));
      EXPECT_EQ(e.left_end, block(ad, 1));
    }
  });
  EXPECT_TRUE(found);
  auto G = triqs_xca::test_utils::ad_to_atom_prop(ad, 2., 40., 1e-10);
  G[root].data() *= 1e-18;
  bs::BlockDiagOpFun native(G);
  EXPECT_EQ(native.get_zero_block_index(root), -1);
  bs::DiagramEvaluator cpu(poles, coeff, G[0].mesh(), ad);
  auto tiny = cpu.compute_single_ptcle_gf(G, topology, index);
  EXPECT_GT(nda::max_element(nda::abs(tiny(nda::range::all, 1, 1))), 1e-24);
  G[root].data() = 0.;
  auto zero      = cpu.compute_single_ptcle_gf(G, topology, index);
  EXPECT_EQ(nda::max_element(nda::abs(zero(nda::range::all, 1, 1))), 0.);
}
TEST(CorrelatorPath, RectangularSidesHaveOneFunctionCapacity) {
  auto ad    = atom(3, true);
  auto model = gpu::export_model(ad, coefficients(3));
  nda::vector<double> poles{1.3, -.8};
  std::vector<bool> live(model.dimensions().size(), true);
  auto mu = fields(model, false), kap = fields(model, true);
  bool rectangular = false;
  for (auto topology : {nda::array<int, 2>{{0, 2}, {1, 3}}, nda::array<int, 2>{{0, 3}, {1, 4}, {2, 5}}, nda::array<int, 2>{{0, 1}, {2, 3}},
                        nda::array<int, 2>{{0, 3}, {1, 2}}}) {
    CorrelatorBackbone backbone(topology, 3);
    long total = backbone.num_flat_indices(poles.size());
    for (long i : {0L, total / 2, total - 1}) {
      backbone.set_flat_index(i, poles);
      auto fragments = gpu::collect_correlator_fragments(model, backbone, live);
      for (auto const *side : {&fragments.right, &fragments.left}) {
        std::size_t bytes = 0;
        for (auto const &p : *side)
          if (p) {
            bytes = gpu::checked_add(bytes, gpu::tensor_bytes(7, p->dimensions.back(), p->dimensions.front()));
            rectangular |= p->dimensions.front() != p->dimensions.back();
          }
        EXPECT_LE(bytes, gpu::function_bytes(7, model.dimensions()));
      }
      auto counts = gpu::for_each_correlator_endpoint(model, fragments, live, mu, kap, [](auto const &, auto const &, auto const &) {});
      EXPECT_EQ(counts.proposed, counts.accepted + counts.pruned);
    }
  }
  EXPECT_TRUE(rectangular);
  CorrelatorBackbone wrong(nda::array<int, 2>{{0, 1}}, 2);
  wrong.set_flat_index(0, poles);
  EXPECT_THROW(gpu::collect_correlator_fragments(model, wrong, live), std::invalid_argument);
}

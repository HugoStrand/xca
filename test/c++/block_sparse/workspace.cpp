#include "test_utils/evaluators.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <tuple>
#include <triqs_xca/block_sparse/backbone_paths.hpp>
#include <triqs_xca/block_sparse/correlator_paths.hpp>
#include <triqs_xca/block_sparse/self_energy_paths.hpp>

namespace {
  namespace logical  = triqs_xca::diagram;
  namespace sparse   = triqs_xca::block_sparse;
  namespace fixtures = triqs_xca::test_utils;
  using nda::dcomplex;
  using Tensor       = nda::array<dcomplex, 3>;
  constexpr auto all = nda::range::all;

  void expect_close(Tensor const &actual, Tensor const &expected) {
    ASSERT_EQ(actual.shape(), expected.shape());
    double scale = nda::max_element(nda::abs(expected));
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 2e-11 * std::max(scale, 1e-10));
  }

  sparse::BlockOp identity(sparse::BlockDiagOpFun const &G) {
    nda::vector<int> destinations(G.get_num_block_cols());
    std::vector<nda::array<dcomplex, 2>> blocks;
    for (int b = 0; b < G.get_num_block_cols(); ++b) {
      destinations(b) = b;
      int d           = G.get_block_size(b);
      auto block      = nda::zeros<dcomplex>(d, d);
      for (int i = 0; i < d; ++i) block(i, i) = 1.;
      blocks.push_back(block);
    }
    return {destinations, blocks};
  }
} // namespace

TEST(BlockSparseWorkspace, changing_dimensions_endpoint_slots_and_retained_forward_segment) {
  auto model = fixtures::build_eval_model("n_partition_spin_xyz");
  sparse::DiagramEvaluator evaluator(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
  triqs_xca::dense::DiagramEvaluator dense(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad_flat, model.dynint_ops, model.dynint_coeffs);
  sparse::BlockDiagOpFun G(fixtures::subspace_blocks(model.G, model.ad, model.mesh));
  sparse::OperatorSupportTable support(evaluator.action.operators);
  auto [mu, kap]  = fixtures::make_correlator_ops(evaluator.action.operators, evaluator.action.n_hyb);
  auto dense_mu   = dense.action.operators.Fs(nda::range(evaluator.action.n_hyb), all, all);
  auto dense_kap  = dense.action.operators.F_dags(nda::range(evaluator.action.n_hyb), all, all);
  auto statistics = logical::uniform_correlator_statistics(true);
  nda::array<int, 2> topology{{0, 3}, {1, 4}, {2, 5}};
  bool grew = false, shrank = false, changed_anchor = false, multiple_endpoint_slots = false;
  bool forward_changed = false, backward_changed = false;
  int productive = 0;
  ASSERT_EQ(nda::max_element(evaluator.action.block_dims), 6);

  for (auto first : {logical::Direction::forward, logical::Direction::backward})
    for (auto second : {logical::Direction::forward, logical::Direction::backward}) {
      for (auto endpoint : {logical::Direction::forward, logical::Direction::backward}) {
        auto backbone = logical::prepare_backbone<logical::Kind::self_energy>(
           topology,
           logical::prepare_arcs(logical::Arcs{{endpoint, -1, -1}, {first, 0, 0}, {second, 1, 1}}, evaluator.action, logical::Kind::self_energy));
        sparse::visit_self_energy_paths(G, backbone, evaluator.action.operators, support, [&](auto const &path) {
          for (int v = 0; v < path.range.vertex_end; ++v) {
            grew |= path.dimension(v) < path.dimension(v + 1);
            shrank |= path.dimension(v) > path.dimension(v + 1);
          }
          changed_anchor |= path.dimension(0) != path.dimension(1);
          multiple_endpoint_slots |= path.zero_operators.size() > 1;
        });
        auto expected = dense.compute_self_energy(model.G, backbone);
        auto result   = evaluator.compute_self_energy(G, backbone);
        auto retained = fixtures::get_tensor_in_full_hilbert_space(result, model.ad, model.mesh.size());
        expect_close(retained, expected);
        productive += nda::max_element(nda::abs(expected)) > 1e-12;
        auto changed_G = G;
        changed_G *= 1.03;
        evaluator.compute_self_energy(changed_G, backbone);
        expect_close(fixtures::get_tensor_in_full_hilbert_space(result, model.ad, model.mesh.size()), retained);
      }

      auto backbone = logical::prepare_backbone<logical::Kind::correlator>(
         topology,
         logical::prepare_arcs(logical::Arcs{{logical::Direction::external, -1, -1}, {first, 0, 0}, {second, 1, 1}}, evaluator.action,
                               logical::Kind::correlator));
      auto expected = dense.compute_one_time_correlator(model.G, backbone, dense_mu, dense_kap, statistics);
      expect_close(evaluator.compute_one_time_correlator(G, backbone, mu, kap, statistics), expected);
      std::vector<sparse::diagram::CorrelatorBackbonePath> before, after;
      auto ranges = backbone.endpoint_ranges();
      sparse::visit_backbone_paths(G, backbone, evaluator.action.operators, support, ranges.before_special, [&](auto const &path) {
        for (int v = path.range.first_vertex; v < path.range.vertex_end; ++v) forward_changed |= path.dimension(v) != path.dimension(v + 1);
        before.push_back(path);
      });
      sparse::visit_backbone_paths(G, backbone, evaluator.action.operators, support, ranges.after_special, [&](auto const &path) {
        for (int v = path.range.first_vertex; v < path.range.vertex_end; ++v) backward_changed |= path.dimension(v) != path.dimension(v + 1);
        after.push_back(path);
      });
      auto sum = nda::zeros<dcomplex>(expected.shape());
      sparse::visit_correlator_paths(G, before, after, mu, kap, [&](auto const &path, int, int) {
        // Complete-path evaluation must keep the forward packed view alive while
        // the backward segment alternately grows and shrinks in U/GKt.
        sum += evaluator.compute_one_time_correlator(G, path, mu, kap, statistics);
      });
      expect_close(sum, expected);
    }
  EXPECT_GT(productive, 0);
  EXPECT_TRUE(grew && shrank && changed_anchor && multiple_endpoint_slots);
  EXPECT_TRUE(forward_changed && backward_changed);
}

TEST(BlockSparseWorkspace, legacy_unknown_sector_capacity_grows_and_retains_results) {
  double beta = 2.;
  auto mesh   = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, 20., 1e-12, false);
  nda::vector<double> poles{0.};
  auto coefficients = nda::ones<dcomplex>(1, 1, 1);
  // The interaction only sees sector zero; external observables may still use
  // sector one, whose dimension cannot be inferred from this operator list.
  nda::vector<int> destinations{0, -1};
  std::vector<Tensor> blocks{nda::ones<dcomplex>(1, 1, 1), nda::zeros<dcomplex>(1, 1, 1)};
  sparse::BlockOpSymSet op(destinations, blocks);
  sparse::BlockOpSymQuartet operators({op}, {op}, coefficients, nda::vector<long>{0});
  for (long dimension : {46341L, std::numeric_limits<long>::max()}) {
    auto oversized = sparse::make_action(mesh, poles, coefficients, operators, {triqs_xca::Statistics::bosonic}, 0, nda::vector<long>{1, dimension});
    EXPECT_THROW((sparse::DiagramEvaluator(std::move(oversized))), std::length_error);
  }
  sparse::DiagramEvaluator evaluator(sparse::make_action(mesh, poles, coefficients, operators, 1));
  ASSERT_EQ(evaluator.action.block_dims(0), 1);
  ASSERT_EQ(evaluator.action.block_dims(1), -1);
  // Explicit numerical construction may leave every dimension unspecified.
  // A valid nonempty operator quartet still supplies the operator count.
  sparse::DiagramEvaluator all_unknown(
     sparse::make_action(mesh, poles, coefficients, operators, {triqs_xca::Statistics::bosonic}, 0, nda::vector<long>{-1, -1}));
  EXPECT_EQ(nda::max_element(all_unknown.action.block_dims), -1);
  auto backbone   = logical::prepare_backbone<logical::Kind::correlator>(nda::array<int, 2>{{0, 1}}, evaluator.action, 0);
  auto statistics = logical::uniform_correlator_statistics(false);
  Tensor retained, original;
  for (int dimension : {3, 7, 2, 9, 1}) {
    SCOPED_TRACE(dimension);
    std::vector<Tensor> values{nda::ones<dcomplex>(mesh.size(), 1, 1), nda::zeros<dcomplex>(mesh.size(), dimension, dimension)};
    for (int t = 0; t < mesh.size(); ++t)
      for (int i = 0; i < dimension; ++i) values[1](t, i, i) = 2.;
    sparse::BlockDiagOpFun G(values, nda::vector<int>{0, 0});
    auto id     = identity(G);
    auto result = evaluator.compute_one_time_correlator(G, backbone, {id}, {id}, statistics);
    expect_close(all_unknown.compute_one_time_correlator(G, backbone, {id}, {id}, statistics), result);
    EXPECT_LE(nda::max_element(nda::abs(result - (1. + 4. * dimension))), 2e-10 * (1. + 4. * dimension));
    sparse::diagram::CorrelatorBackbonePath path(backbone);
    for (int v = path.range.first_vertex; v <= path.range.vertex_end; ++v) {
      path.sector(v)    = 1;
      path.dimension(v) = dimension;
    }
    path.zero_operators = path.special_operators = {0};
    auto contribution                            = evaluator.compute_one_time_correlator(G, path, {id}, {id}, statistics);
    EXPECT_LE(nda::max_element(nda::abs(contribution - 4. * dimension)), 2e-10 * 4. * dimension);
    if (dimension == 3) {
      retained = std::move(result);
      original = retained;
    }
    expect_close(retained, original);
  }

  auto sigma_backbone = logical::prepare_backbone<logical::Kind::self_energy>(nda::array<int, 2>{{0, 1}}, evaluator.action, 0);
  for (auto shape : {std::array<long, 3>{mesh.size(), 3, 2}, std::array<long, 3>{mesh.size(), 0, 0}}) {
    std::vector<Tensor> values{nda::ones<dcomplex>(mesh.size(), 1, 1), nda::zeros<dcomplex>(shape)};
    sparse::BlockDiagOpFun wrong(values, nda::vector<int>{0, 0});
    EXPECT_THROW(evaluator.compute_self_energy(wrong, sigma_backbone), std::invalid_argument);
    EXPECT_THROW(evaluator.compute_one_time_correlator(wrong, backbone, {}, {}, statistics), std::invalid_argument);
  }
}

TEST(BlockSparseWorkspace, copied_evaluator_outlives_source_action_and_mesh) {
  auto [evaluator, G, backbone, expected] = [] {
    auto model = fixtures::build_eval_model("unequal_sets_nA0");
    sparse::DiagramEvaluator original(model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs);
    sparse::BlockDiagOpFun G(fixtures::subspace_blocks(model.G, model.ad, model.mesh));
    auto backbone = logical::prepare_backbone<logical::Kind::self_energy>(
       nda::array<int, 2>{{0, 2}, {1, 3}},
       logical::prepare_arcs(logical::Arcs{{logical::Direction::forward, -1, -1}, {logical::Direction::forward, original.action.n_flavours() - 1, 1}},
                             original.action, logical::Kind::self_energy));
    auto expected = original.compute_self_energy(G, backbone);
    auto copy     = std::make_unique<sparse::DiagramEvaluator>(original);
    return std::tuple{std::move(copy), std::move(G), std::move(backbone), std::move(expected)};
  }();
  // The original evaluator, atom_diag and source mesh are all gone. Its copy
  // keeps the Action's shared interpolation object and independent scratch alive.
  double scale = 0.;
  for (int b = 0; b < G.get_num_block_cols(); ++b) scale = std::max(scale, nda::max_element(nda::abs(expected.get_block(b))));
  ASSERT_GT(scale, 1e-12);
  auto result  = evaluator->compute_self_energy(G, backbone);
  auto changed = G;
  changed *= 1.05;
  auto changed_result = evaluator->compute_self_energy(changed, backbone);
  for (int b = 0; b < G.get_num_block_cols(); ++b) {
    expect_close(nda::make_regular(result.get_block(b)), nda::make_regular(expected.get_block(b)));
    expect_close(nda::make_regular(changed_result.get_block(b)), nda::make_regular((1.05 * 1.05 * 1.05) * expected.get_block(b)));
  }
}

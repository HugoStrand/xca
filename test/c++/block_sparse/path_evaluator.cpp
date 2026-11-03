#include "test_utils/prepared_evaluators.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <triqs_xca/block_sparse/backbone_paths.hpp>
#include <triqs_xca/block_sparse/correlator_paths.hpp>
#include <triqs_xca/block_sparse/self_energy_paths.hpp>
#include "test_utils/evaluators.hpp"

namespace {
  namespace logical = triqs_xca::diagram;
  namespace sparse  = triqs_xca::block_sparse;
  using SelfEnergyPath = sparse::diagram::SelfEnergyBackbonePath;
  using CorrelatorPath = sparse::diagram::CorrelatorBackbonePath;
  using tensor      = nda::array<nda::dcomplex, 3>;
  using nda::dcomplex;

  template <typename Path>
  concept SelfEnergyInput = requires(sparse::DiagramEvaluator &evaluator, sparse::BlockDiagOpFun const &G, Path const &path) {
    evaluator.compute_self_energy(G, path);
  };

  template <typename Path>
  concept CorrelatorInput = requires(sparse::DiagramEvaluator &evaluator, sparse::BlockDiagOpFun const &G, Path const &path,
                                     std::vector<sparse::BlockOp> const &operators) {
    evaluator.compute_one_time_correlator(G, path, operators, operators, logical::CorrelatorStatistics{});
  };

  static_assert(SelfEnergyInput<SelfEnergyPath>);
  static_assert(!SelfEnergyInput<CorrelatorPath>);
  static_assert(CorrelatorInput<CorrelatorPath>);
  static_assert(!CorrelatorInput<SelfEnergyPath>);

  void expect_close(tensor const &actual, tensor const &expected) {
    double scale = nda::max_element(nda::abs(expected));
    ASSERT_GT(scale, 1e-14) << "the selected contribution must be resolved";
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 1e-12 * scale);
  }

  sparse::BlockOp identity(sparse::BlockDiagOpFun const &G) {
    nda::vector<int> destinations(G.get_num_block_cols());
    std::vector<nda::array<dcomplex, 2>> blocks;
    for (int b = 0; b < G.get_num_block_cols(); ++b) {
      destinations(b) = b;
      int d            = G.get_block_size(b);
      auto block       = nda::zeros<dcomplex>(d, d);
      for (int i = 0; i < d; ++i) block(i, i) = 1.;
      blocks.emplace_back(block);
    }
    return {destinations, blocks};
  }

  sparse::BlockOp physical_operator(sparse::BlockOpSymQuartet const &operators, int flavour, bool adjoint) {
    int group = static_cast<int>(operators.sym_set_labels(flavour));
    int index = static_cast<int>(operators.sym_set_inds(flavour));
    auto const &op = adjoint ? operators.F_dags[group] : operators.Fs[group];
    nda::vector<int> destinations = op.get_block_indices();
    std::vector<nda::array<dcomplex, 2>> blocks;
    for (int b = 0; b < op.get_num_block_cols(); ++b) {
      if (op.get_block_index(b) == -1)
        blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      else
        blocks.emplace_back(op.get_block(b)(index, nda::range::all, nda::range::all));
    }
    return {destinations, blocks};
  }

  nda::array<int, 2> topology_for_order(int order) {
    return order == 1 ? nda::array<int, 2>{{0, 1}} : nda::array<int, 2>{{0, 2}, {1, 3}};
  }

  class BlockSparsePathEvaluator : public ::testing::Test {
    protected:
    triqs_xca::test_utils::EvalModel model = triqs_xca::test_utils::build_eval_model("unequal_sets_nA0");
    sparse::DiagramEvaluator evaluator{model.hyb_poles, model.hyb_coeffs, model.mesh, model.ad, model.dynint_ops, model.dynint_coeffs};
    sparse::BlockDiagOpFun G{triqs_xca::test_utils::subspace_blocks(model.G, model.ad, model.mesh)};
    sparse::BlockOp id = identity(G);
    std::vector<sparse::BlockOp> mu{id, dcomplex{0.7, 0.2} * id}, kap{id, dcomplex{-0.3, 0.4} * id};
    logical::CorrelatorStatistics statistics{triqs_xca::Statistics::bosonic, triqs_xca::Statistics::bosonic};

    tensor self_energy(sparse::BlockDiagOpFun const &propagator, SelfEnergyPath const &path) {
      return triqs_xca::test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(propagator, path), model.ad, model.mesh.size());
    }

    std::vector<CorrelatorPath> correlator_paths(logical::CorrelatorBackbone const &backbone, bool &reused_segments) {
      sparse::OperatorSupportTable support(evaluator.action.operators);
      std::vector<CorrelatorPath> before, after, completed;
      auto ranges = backbone.endpoint_ranges();
      sparse::visit_backbone_paths(G, backbone, evaluator.action.operators, support, ranges.before_special,
                                   [&](auto const &path) { before.push_back(path); });
      sparse::visit_backbone_paths(G, backbone, evaluator.action.operators, support, ranges.after_special,
                                   [&](auto const &path) { after.push_back(path); });
      std::map<std::pair<int, int>, int> first_mu;
      sparse::visit_correlator_paths(G, before, after, mu, kap, [&](auto const &path, int before_result, int after_result) {
        ASSERT_EQ(path.special_operators.size(), 1);
        auto [entry, inserted] = first_mu.emplace(std::pair{before_result, after_result}, path.special_operators[0]);
        if (!inserted && entry->second != path.special_operators[0]) reused_segments = true;
        completed.push_back(path);
      });
      // Neither discovery workspaces nor segment tables escape this helper.
      return completed;
    }
  };
} // namespace

TEST_F(BlockSparsePathEvaluator, retained_self_energy_paths_match_prepared_backbones_with_changed_propagator) {
  for (int order : {1, 2}) {
    for (auto direction : {logical::Direction::backward, logical::Direction::forward}) {
      SCOPED_TRACE(::testing::Message() << "order=" << order << ", direction=" << static_cast<int>(direction));
      tensor expected;
      auto paths = [&] {
        auto topology = topology_for_order(order);
        // The internal arc is forward, with the density flavour and negative pole.
        long f = order == 1 ? static_cast<int>(direction)
                            : (evaluator.action.n_flavours() - 1) + evaluator.action.n_flavours() * (1 + model.hyb_poles.size() * (2 + static_cast<int>(direction)));
        logical::Arcs choices{{direction, -1, -1}};
        if (order == 2) choices.push_back({logical::Direction::forward, evaluator.action.n_flavours() - 1, 1});
        auto arcs = logical::prepare_arcs(std::move(choices), evaluator.action, logical::Kind::self_energy);
        auto backbone = logical::prepare_backbone<logical::Kind::self_energy>(topology, arcs);
        expected = triqs_xca::test_utils::get_tensor_in_full_hilbert_space(triqs_xca::test_utils::fixture_self_energy(evaluator, G, topology, f), model.ad);
        expect_close(triqs_xca::test_utils::get_tensor_in_full_hilbert_space(evaluator.compute_self_energy(G, backbone), model.ad), expected);
        if (order == 2) EXPECT_GT(std::abs(triqs_xca::tau_kernel_prefactor(backbone.kernel_prefactor, evaluator.action.kernel) - 1.), 0.1);
        sparse::OperatorSupportTable support(evaluator.action.operators);
        std::vector<SelfEnergyPath> retained;
        sparse::visit_self_energy_paths(G, backbone, evaluator.action.operators, support,
                                        [&](auto const &path) { retained.push_back(path); });
        return retained;
      }();
      ASSERT_GT(paths.size(), 1);
      // The topology, arcs, source backbone and discovery workspace are gone.
      tensor actual(expected.shape());
      actual = 0.;
      for (auto const &path : paths) actual += self_energy(G, path);
      expect_close(actual, expected);

      auto changed_G = G;
      changed_G *= 1.1;
      actual = 0.;
      for (auto const &path : paths) actual += self_energy(changed_G, path);
      expect_close(actual, nda::make_regular(std::pow(1.1, 2 * order - 1) * expected));
    }
  }
}

TEST_F(BlockSparsePathEvaluator, retained_correlator_paths_do_not_depend_on_segment_result_caches) {
  for (int order : {1, 2}) {
    SCOPED_TRACE(::testing::Message() << "order=" << order);
    tensor expected;
    bool reused_segments = false;
    auto paths = [&] {
      auto topology = topology_for_order(order);
      long f = order == 1 ? 0 : (evaluator.action.n_flavours() - 1) + evaluator.action.n_flavours() * (1 + model.hyb_poles.size());
      logical::Arcs choices{{logical::Direction::external, -1, -1}};
      if (order == 2) choices.push_back({logical::Direction::forward, evaluator.action.n_flavours() - 1, 1});
      auto arcs = logical::prepare_arcs(std::move(choices), evaluator.action, logical::Kind::correlator);
      auto backbone = logical::prepare_backbone<logical::Kind::correlator>(topology, arcs);
      expected = triqs_xca::test_utils::fixture_correlator(evaluator, G, topology, mu, kap, f, false);
      // This call evaluates and then releases its numerical segment-result caches.
      expect_close(evaluator.compute_one_time_correlator(G, backbone, mu, kap, statistics), expected);
      if (order == 1) {
        auto ranges = backbone.endpoint_ranges();
        EXPECT_EQ(ranges.before_special.first_vertex, ranges.before_special.vertex_end);
        EXPECT_EQ(ranges.after_special.first_vertex, ranges.after_special.vertex_end);
      } else {
        EXPECT_GT(std::abs(triqs_xca::tau_kernel_prefactor(backbone.kernel_prefactor, evaluator.action.kernel) - 1.), 0.1);
      }
      return correlator_paths(backbone, reused_segments);
    }();
    ASSERT_GT(paths.size(), 1);
    ASSERT_TRUE(reused_segments) << "distinct observable joins must reuse the same pair of segment results";
    tensor actual(expected.shape());
    actual = 0.;
    for (auto const &path : paths) {
      ASSERT_EQ(path.special_operators.size(), 1);
      EXPECT_EQ(path.zero_operators, (std::vector<int>{0, 1}));
      actual += evaluator.compute_one_time_correlator(G, path, mu, kap, statistics);
    }
    expect_close(actual, expected);
    for (int mu_index = 0; mu_index < 2; ++mu_index)
      for (int kap_index = 0; kap_index < 2; ++kap_index)
        EXPECT_GT(nda::max_element(nda::abs(expected(nda::range::all, mu_index, kap_index))), 1e-14);

    auto changed_G = G;
    changed_G *= 1.1;
    actual = 0.;
    for (auto const &path : paths) actual += evaluator.compute_one_time_correlator(changed_G, path, mu, kap, statistics);
    expect_close(actual, nda::make_regular(std::pow(1.1, 2 * order) * expected));
  }
}

TEST_F(BlockSparsePathEvaluator, retained_correlator_paths_preserve_fermionic_crossing_sign) {
  mu.clear();
  kap.clear();
  for (int flavour = 0; flavour < evaluator.action.n_hyb; ++flavour) {
    mu.push_back(physical_operator(evaluator.action.operators, flavour, false));
    kap.push_back(physical_operator(evaluator.action.operators, flavour, true));
  }
  statistics = {triqs_xca::Statistics::fermionic, triqs_xca::Statistics::fermionic};
  tensor expected;
  bool reused_segments = false;
  auto paths = [&] {
    auto topology = topology_for_order(2);
    // Crossing a fermionic internal arc with the external fermionic pair has odd parity.
    long f = evaluator.action.n_flavours() * (1 + model.hyb_poles.size());
    auto arcs = logical::prepare_arcs(logical::Arcs{{logical::Direction::external, -1, -1}, {logical::Direction::forward, 0, 1}},
                                       evaluator.action, logical::Kind::correlator);
    auto backbone = logical::prepare_backbone<logical::Kind::correlator>(topology, arcs);
    EXPECT_EQ(backbone.endpoints.signs.with_fermionic_arc, -1);
    EXPECT_EQ(backbone.endpoints.signs.without_fermionic_arc, 1);
    expected = triqs_xca::test_utils::fixture_correlator(evaluator, G, topology, mu, kap, f, true);
    expect_close(evaluator.compute_one_time_correlator(G, backbone, mu, kap, statistics), expected);
    return correlator_paths(backbone, reused_segments);
  }();
  ASSERT_FALSE(paths.empty());
  tensor actual(expected.shape());
  actual = 0.;
  for (auto const &path : paths) actual += evaluator.compute_one_time_correlator(G, path, mu, kap, statistics);
  expect_close(actual, expected);

  auto changed_G = G;
  changed_G *= 1.1;
  actual = 0.;
  for (auto const &path : paths) actual += evaluator.compute_one_time_correlator(changed_G, path, mu, kap, statistics);
  expect_close(actual, nda::make_regular(std::pow(1.1, 4) * expected));
}

TEST_F(BlockSparsePathEvaluator, path_entry_points_reject_incomplete_operations_and_honor_empty_selections) {
  auto topology = topology_for_order(1);
  auto sigma = logical::prepare_backbone<logical::Kind::self_energy>(topology, logical::decode_arcs(1, evaluator.action, 0, logical::Kind::self_energy));
  auto corr = logical::prepare_backbone<logical::Kind::correlator>(topology, logical::decode_arcs(1, evaluator.action, 0, logical::Kind::correlator));
  std::vector<SelfEnergyPath> sigma_paths;
  sparse::visit_self_energy_paths(G, sigma, evaluator.action.operators, sparse::OperatorSupportTable(evaluator.action.operators),
                                  [&](auto const &path) { sigma_paths.push_back(path); });
  bool reused_segments = false;
  auto corr_paths = correlator_paths(corr, reused_segments);
  ASSERT_FALSE(sigma_paths.empty());
  ASSERT_FALSE(corr_paths.empty());
  auto const &sigma_path = sigma_paths.front();
  auto const &corr_path = corr_paths.front();
  auto expected_sigma = self_energy(G, sigma_path);
  auto expected_corr = evaluator.compute_one_time_correlator(G, corr_path, mu, kap, statistics);

  // Retained outputs must survive mixed later operations and rejected paths.
  auto retained_sigma = evaluator.compute_self_energy(G, sigma_path);
  auto retained_corr = evaluator.compute_one_time_correlator(G, corr_path, mu, kap, statistics);
  auto changed_G = G;
  changed_G *= 1.2;
  evaluator.compute_one_time_correlator(changed_G, corr_path, mu, kap, statistics);
  evaluator.compute_self_energy(changed_G, sigma_path);
  SelfEnergyPath sigma_segment(sigma, sigma.endpoint_ranges().before_special);
  CorrelatorPath corr_segment(corr, corr.endpoint_ranges().before_special);
  EXPECT_THROW(evaluator.compute_self_energy(G, sigma_segment), std::invalid_argument);
  EXPECT_THROW(evaluator.compute_one_time_correlator(G, corr_segment, mu, kap, statistics), std::invalid_argument);
  auto two_tau_operators = corr_path;
  two_tau_operators.special_operators = {0, 1};
  two_tau_operators.zero_operators.clear();
  EXPECT_THROW(evaluator.compute_one_time_correlator(G, two_tau_operators, mu, kap, statistics), std::invalid_argument);
  expect_close(triqs_xca::test_utils::get_tensor_in_full_hilbert_space(retained_sigma, model.ad, model.mesh.size()), expected_sigma);
  expect_close(retained_corr, expected_corr);

  for (bool clear_zero : {false, true}) {
    auto empty_sigma = sigma_path;
    auto empty_corr = corr_path;
    (clear_zero ? empty_sigma.zero_operators : empty_sigma.special_operators).clear();
    (clear_zero ? empty_corr.zero_operators : empty_corr.special_operators).clear();
    EXPECT_EQ(nda::max_element(nda::abs(self_energy(G, empty_sigma))), 0.);
    EXPECT_EQ(nda::max_element(nda::abs(evaluator.compute_one_time_correlator(G, empty_corr, mu, kap, statistics))), 0.);
  }
  expect_close(self_energy(G, sigma_path), expected_sigma);
  expect_close(evaluator.compute_one_time_correlator(G, corr_path, mu, kap, statistics), expected_corr);
}

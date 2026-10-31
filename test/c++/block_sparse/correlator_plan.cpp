#include <triqs_xca/block_sparse/backbone_paths.hpp>
#include <triqs_xca/diagram/backbone.hpp>
#include <triqs_xca/block_sparse/correlator_paths.hpp>
#include <gtest/gtest.h>

using triqs_xca::diagram::Direction;

namespace {
  using namespace triqs_xca;
  using namespace triqs_xca::block_sparse;

  struct Ladder {
    nda::vector<int> down{-1, 0, 1}, up{1, 2, -1};
    nda::array<int, 2> down_dims{{0, 1}, {1, 2}, {2, 1}}, up_dims{{2, 1}, {1, 2}, {0, 1}};
    BlockOpSymQuartet make_operators() {
      BlockOpSymSet F(1, down, down_dims), Fdag(1, up, up_dims);
      for (int b = 0; b < 3; ++b) {
        if (down(b) >= 0) F.set_block(b, nda::ones<dcomplex>(1, down_dims(b, 0), down_dims(b, 1)));
        if (up(b) >= 0) Fdag.set_block(b, nda::ones<dcomplex>(1, up_dims(b, 0), up_dims(b, 1)));
      }
      return {{F}, {Fdag}, nda::ones<dcomplex>(1, 1, 1), nda::vector<long>{0}};
    }
    BlockOpSymQuartet operators = make_operators();
    BlockOp raising{up, up_dims}, lowering{down, down_dims};

    BlockDiagOpFun propagator(nda::vector<int> const &mask = {0, 0, 0}) {
      std::vector<nda::array<dcomplex, 3>> blocks;
      for (int d : {1, 2, 1}) blocks.push_back(nda::zeros<dcomplex>(1, d, d));
      return {blocks, mask};
    }
  };

  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> fragments(BlockDiagOpFun const &G, triqs_xca::diagram::Backbone const &assignment,
                                                                        BlockOpSymQuartet const &operators, triqs_xca::diagram::BackboneRange range) {
    std::vector<triqs_xca::block_sparse::diagram::BackbonePath> result;
    triqs_xca::block_sparse::visit_backbone_paths(G, assignment, operators, OperatorSupportTable(operators), range,
                                                  [&](auto const &p) { result.push_back(p); });
    return result;
  }
} // namespace

TEST(CorrelatorPlan, crossing_fragments_have_independent_owned_paths) {
  Ladder model;
  auto G          = model.propagator();
  auto assignment = triqs_xca::diagram::prepare_backbone(
     triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}},
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, 0, 0}},
                                      std::vector<triqs_xca::Statistics>(1, triqs_xca::Statistics::bosonic), nda::vector<double>{1.},
                                      triqs_xca::diagram::Kind::correlator));
  auto right = fragments(G, assignment, model.operators, assignment.endpoint_ranges().before_special);
  auto left  = fragments(G, assignment, model.operators, assignment.endpoint_ranges().after_special);
  ASSERT_EQ(right.size(), 2);
  ASSERT_EQ(left.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(right[i].sectors.size(), 2);
    EXPECT_EQ(left[i].sectors.size(), 2);
    EXPECT_EQ(right[i].range.first_vertex, 1);
    EXPECT_EQ(left[i].range.first_vertex, 3);
    EXPECT_EQ(right[i].range.first_edge, 0);
    EXPECT_EQ((right[i].range.edge_end - 1), 1);
    EXPECT_EQ(right[i].sector(right[i].range.first_vertex), i + 1);
    EXPECT_EQ(right[i].sector(right[i].range.vertex_end), i);
    EXPECT_EQ(right[i].edge_sector(0), i + 1);
    EXPECT_EQ(right[i].edge_sector(1), i);
    EXPECT_EQ(left[i].range.first_edge, 2);
    EXPECT_EQ((left[i].range.edge_end - 1), 3);
    EXPECT_EQ(left[i].sector(left[i].range.first_vertex), i);
    EXPECT_EQ(left[i].sector(left[i].range.vertex_end), i + 1);
    EXPECT_EQ(left[i].edge_sector(2), i);
    EXPECT_EQ(left[i].edge_sector(3), i + 1);
  }
  auto retained              = right[0];
  right[0].sector_dimensions = -99;
  EXPECT_EQ(retained.dimension(1), 2);
  EXPECT_EQ(retained.dimension(2), 1);

  auto masked    = model.propagator({-1, 0, 0});
  auto surviving = fragments(masked, assignment, model.operators, assignment.endpoint_ranges().before_special);
  ASSERT_EQ(surviving.size(), 1);
  EXPECT_EQ(surviving[0].sector(surviving[0].range.first_vertex), 2);
  EXPECT_EQ(surviving[0].sector(surviving[0].range.vertex_end), 1);
}

TEST(CorrelatorPlan, single_edge_fragments_survive_a_failed_walk_on_the_other_side) {
  Ladder model;
  auto G = model.propagator();
  // The [tau,beta] side has no internal vertices. The preceding right walk
  // fails at its last starting sector; that must not poison the single-edge side.
  auto assignment = triqs_xca::diagram::prepare_backbone(
     triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 3}, {1, 2}}},
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::external, -1, -1}, {Direction::backward, 0, 0}},
                                      std::vector<triqs_xca::Statistics>(1, triqs_xca::Statistics::bosonic), nda::vector<double>{1.},
                                      triqs_xca::diagram::Kind::correlator));
  auto right = fragments(G, assignment, model.operators, assignment.endpoint_ranges().before_special);
  auto left  = fragments(G, assignment, model.operators, assignment.endpoint_ranges().after_special);
  ASSERT_EQ(right.size(), 2);
  ASSERT_EQ(left.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(left[i].sectors.size(), 1);
    EXPECT_EQ(left[i].range.first_vertex, left[i].range.vertex_end);
    EXPECT_EQ(left[i].dimension(left[i].range.first_vertex), i == 1 ? 2 : 1);
    EXPECT_EQ(left[i].range.first_edge, (left[i].range.edge_end - 1));
    EXPECT_EQ(left[i].sector(left[i].range.first_vertex), i);
    EXPECT_EQ(left[i].sector(left[i].range.vertex_end), i);
  }
}

TEST(CorrelatorPlan, external_joins_prescribe_closure_and_observable_destinations) {
  Ladder model;
  auto G = model.propagator();
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 1}}};
  auto arcs       = triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::external, -1, -1}},
                                                     std::vector<triqs_xca::Statistics>{triqs_xca::Statistics::bosonic}, nda::vector<double>{1.},
                                                     triqs_xca::diagram::Kind::correlator);
  auto assignment = triqs_xca::diagram::prepare_backbone(topology, arcs);
  auto right      = fragments(G, assignment, model.operators, assignment.endpoint_ranges().before_special);
  auto left       = fragments(G, assignment, model.operators, assignment.endpoint_ranges().after_special);
  ASSERT_EQ(right.size(), 3);
  ASSERT_EQ(left.size(), 3);
  std::vector<BlockOp> mu{model.raising}, kap{model.raising, model.lowering};
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> joins;
  std::vector<std::pair<int, int>> result_indices;
  triqs_xca::block_sparse::visit_correlator_paths(G, right, left, mu, kap, [&](auto const &path, int before, int after) {
    joins.push_back(path);
    result_indices.emplace_back(before, after);
  });
  ASSERT_EQ(joins.size(), 2);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(joins[i].special_operators.size(), 1);
    EXPECT_EQ(joins[i].special_operators[0], 0);
    EXPECT_EQ(result_indices[i].first, i);
    EXPECT_EQ(result_indices[i].second, i + 1);
    ASSERT_EQ(joins[i].zero_operators.size(), 1);
    EXPECT_EQ(joins[i].zero_operators[0], 1);
    EXPECT_EQ(joins[i].sector(joins[i].backbone.endpoints.special_vertex), i);
    EXPECT_EQ(joins[i].sector(0), i + 1);
  }
  auto masked = model.propagator({0, -1, 0});
  int count   = 0;
  triqs_xca::block_sparse::visit_correlator_paths(masked, right, left, mu, kap, [&](auto const &, int, int) { ++count; });
  EXPECT_EQ(count, 0);
  left.erase(left.begin() + 1); // absent middle path; table indices differ from sectors
  joins.clear();
  result_indices.clear();
  triqs_xca::block_sparse::visit_correlator_paths(G, right, left, mu, kap, [&](auto const &path, int before, int after) {
    joins.push_back(path);
    result_indices.emplace_back(before, after);
  });
  ASSERT_EQ(joins.size(), 1);
  EXPECT_EQ(result_indices[0].first, 1);
  EXPECT_EQ(result_indices[0].second, 1); // retained table index, not sector 2
  EXPECT_EQ(joins[0].sector(joins[0].backbone.endpoints.special_vertex), 1);
  EXPECT_EQ(joins[0].sector(0), 2);
}

TEST(CorrelatorPlan, discovery_accepts_backbone_ranges_without_correlator_side_labels) {
  Ladder model;
  auto G          = model.propagator();
  auto assignment = triqs_xca::diagram::prepare_backbone(
     triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}},
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, 0, 0}},
                                      std::vector<triqs_xca::Statistics>(1, triqs_xca::Statistics::bosonic), nda::vector<double>{1.},
                                      triqs_xca::diagram::Kind::correlator));
  // Leading selected vertex 1 followed by edge 1: neither complete correlator half.
  auto paths = fragments(G, assignment, model.operators, {1, 2, 1, 2});
  ASSERT_EQ(paths.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(paths[i].sector(1), i + 1);
    EXPECT_EQ(paths[i].sector(2), i);
    EXPECT_EQ(paths[i].edge_sector(1), i);
  }
  auto edge_only = fragments(G, assignment, model.operators, {2, 2, 1, 2});
  ASSERT_EQ(edge_only.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(edge_only[i].sectors.size(), 1);
    EXPECT_EQ(edge_only[i].sector(2), i);
  }
  EXPECT_THROW(fragments(G, assignment, model.operators, {3, 5, 2, 5}), std::invalid_argument);
  EXPECT_THROW(fragments(G, assignment, model.operators, {0, 2, 0, 2}), std::invalid_argument); // external operator unresolved
}

TEST(CorrelatorPlan, empty_range_has_no_independent_path_contribution) {
  Ladder model;
  auto G = model.propagator();
  triqs_xca::diagram::Topology topology{nda::array<int, 2>{{0, 1}}};
  auto arcs       = triqs_xca::diagram::decode_arcs(topology.order(), std::vector<triqs_xca::Statistics>{triqs_xca::Statistics::bosonic},
                                                    nda::vector<double>{1.}, 0, triqs_xca::diagram::Kind::self_energy);
  auto assignment = triqs_xca::diagram::prepare_backbone(topology, arcs);
  auto tail       = assignment.endpoint_ranges().after_special;
  ASSERT_TRUE(tail.empty());
  EXPECT_TRUE(fragments(G, assignment, model.operators, tail).empty());
}

TEST(CorrelatorPlan, complete_paths_own_all_slots_and_endpoint_choices_after_tables_disappear) {
  Ladder model;
  auto G        = model.propagator();
  auto backbone = triqs_xca::diagram::prepare_backbone(
     triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}},
     triqs_xca::diagram::prepare_arcs(triqs_xca::diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, 0, 0}},
                                      std::vector<triqs_xca::Statistics>{triqs_xca::Statistics::bosonic}, nda::vector<double>{1.},
                                      triqs_xca::diagram::Kind::correlator));
  auto right = fragments(G, backbone, model.operators, backbone.endpoint_ranges().before_special);
  auto left  = fragments(G, backbone, model.operators, backbone.endpoint_ranges().after_special);
  std::vector<BlockOp> mu{model.raising, model.lowering}, kap{model.lowering, model.raising};
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> retained;
  std::vector<std::pair<int, int>> result_indices;
  triqs_xca::block_sparse::visit_correlator_paths(G, right, left, mu, kap, [&](auto const &path, int before, int after) {
    retained.push_back(path);
    result_indices.emplace_back(before, after);
  });
  ASSERT_EQ(retained.size(), 2);
  EXPECT_EQ(result_indices, (std::vector<std::pair<int, int>>{{0, 1}, {1, 0}}));
  backbone.vertices[1].flavour = -99;
  right.clear();
  left.clear();
  for (int i = 0; i < 2; ++i) {
    auto const &path = retained[i];
    EXPECT_EQ(path.range, path.backbone.range());
    EXPECT_EQ(path.backbone.vertices[1].flavour, 0);
    EXPECT_EQ(path.special_operators, (std::vector<int>{i}));
    EXPECT_EQ(path.zero_operators, (std::vector<int>{i}));
  }
  EXPECT_EQ((std::vector<int>{retained[0].sectors.begin(), retained[0].sectors.end()}), (std::vector<int>{2, 1, 0, 1, 2}));
  EXPECT_EQ((std::vector<int>{retained[0].sector_dimensions.begin(), retained[0].sector_dimensions.end()}), (std::vector<int>{1, 2, 1, 2, 1}));
  EXPECT_EQ((std::vector<int>{retained[1].sectors.begin(), retained[1].sectors.end()}), (std::vector<int>{1, 2, 1, 0, 1}));
  EXPECT_EQ((std::vector<int>{retained[1].sector_dimensions.begin(), retained[1].sector_dimensions.end()}), (std::vector<int>{2, 1, 2, 1, 2}));
  auto copy = retained[0];
  copy.backbone.edges[1].kernels.clear();
  copy.zero_operators.clear();
  copy.sector(0) = -1;
  EXPECT_FALSE(retained[0].backbone.edges[1].kernels.empty());
  EXPECT_EQ(retained[0].zero_operators, (std::vector<int>{0}));
  EXPECT_EQ(retained[0].sector(0), 2);
  // A missing table has no completion, even when a retained complete path exists.
  int visits = 0;
  triqs_xca::block_sparse::visit_correlator_paths(G, right, retained, mu, kap, [&](auto const &, int, int) { ++visits; });
  triqs_xca::block_sparse::visit_correlator_paths(G, retained, left, mu, kap, [&](auto const &, int, int) { ++visits; });
  EXPECT_EQ(visits, 0);
}

#include "triqs_xca/block_sparse/bind/backbone.hpp"
#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/bind/correlator.hpp>

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

std::vector<triqs_xca::block_sparse::diagram::BackbonePath> fragments(BlockDiagOpFun const &G, triqs_xca::diagram::Diagram const &assignment,
                                         BlockOpSymQuartet const &operators, triqs_xca::diagram::CorrelatorSegment side) {
  std::vector<triqs_xca::block_sparse::diagram::BackbonePath> result;
  triqs_xca::block_sparse::diagram::visit_correlator_paths(G, assignment, operators, OperatorSupportTable(operators), side, [&](auto const &p) { result.push_back(p); });
  return result;
}
}

TEST(CorrelatorPlan, crossing_fragments_have_independent_owned_paths) {
  Ladder model;
  auto G = model.propagator();
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}}, 1,
                                       triqs_xca::diagram::Indices{{1}, {0}, {0}}, nda::vector<double>{1.}, triqs_xca::diagram::Kind::correlator);
  auto right = fragments(G, assignment, model.operators, triqs_xca::diagram::CorrelatorSegment::zero_to_tau);
  auto left = fragments(G, assignment, model.operators, triqs_xca::diagram::CorrelatorSegment::tau_to_beta);
  ASSERT_EQ(right.size(), 2);
  ASSERT_EQ(left.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(right[i].sectors.size(), 2);
    EXPECT_EQ(left[i].sectors.size(), 2);
    EXPECT_EQ(right[i].first_vertex, 1);
    EXPECT_EQ(left[i].first_vertex, 3);
    EXPECT_EQ(right[i].first_edge, 0);
    EXPECT_EQ((right[i].edge_end - 1), 1);
    EXPECT_EQ(right[i].sector(right[i].first_vertex), i + 1);
    EXPECT_EQ(right[i].sector(right[i].vertex_end), i);
    EXPECT_EQ(right[i].edge_sector(0), i + 1);
    EXPECT_EQ(right[i].edge_sector(1), i);
    EXPECT_EQ(left[i].first_edge, 2);
    EXPECT_EQ((left[i].edge_end - 1), 3);
    EXPECT_EQ(left[i].sector(left[i].first_vertex), i);
    EXPECT_EQ(left[i].sector(left[i].vertex_end), i + 1);
    EXPECT_EQ(left[i].edge_sector(2), i);
    EXPECT_EQ(left[i].edge_sector(3), i + 1);
  }
  auto bound = triqs_xca::block_sparse::bind::backbone(G, assignment, model.operators, right[0]);
  EXPECT_EQ(bound.vertices.size(), 1);
  EXPECT_EQ(bound.propagators.size(), 2);
  EXPECT_EQ(bound.propagator(0).data(), G.get_block(1).data());
  auto beta_bound = triqs_xca::block_sparse::bind::backbone(G, assignment, model.operators, left[0]);
  EXPECT_EQ(beta_bound.vertices.size(), 1);
  EXPECT_EQ(beta_bound.propagators.size(), 2);
  EXPECT_EQ(beta_bound.propagator(3).data(), G.get_block(1).data());
  EXPECT_EQ(bound.vertex(1).extent(0), 1);
  EXPECT_EQ(bound.vertex(1).extent(1), 2);
  right[0].sector_dimensions = -99;
  EXPECT_EQ(bound.path.dimension(1), 2);
  EXPECT_EQ(bound.path.dimension(2), 1);

  auto masked = model.propagator({-1, 0, 0});
  auto surviving = fragments(masked, assignment, model.operators, triqs_xca::diagram::CorrelatorSegment::zero_to_tau);
  ASSERT_EQ(surviving.size(), 1);
  EXPECT_EQ(surviving[0].sector(surviving[0].first_vertex), 2);
  EXPECT_EQ(surviving[0].sector(surviving[0].vertex_end), 1);
}

TEST(CorrelatorPlan, empty_fragments_survive_a_failed_walk_on_the_other_side) {
  Ladder model;
  auto G = model.propagator();
  // The [tau,beta] side has no internal vertices. The preceding right walk
  // fails at its last starting sector; that must not poison the empty side.
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 3}, {1, 2}}}, 1,
                                       triqs_xca::diagram::Indices{{0}, {0}, {0}}, nda::vector<double>{1.}, triqs_xca::diagram::Kind::correlator);
  auto right = fragments(G, assignment, model.operators, triqs_xca::diagram::CorrelatorSegment::zero_to_tau);
  auto left = fragments(G, assignment, model.operators, triqs_xca::diagram::CorrelatorSegment::tau_to_beta);
  ASSERT_EQ(right.size(), 2);
  ASSERT_EQ(left.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(left[i].sectors.size(), 1);
    EXPECT_EQ(left[i].first_vertex, left[i].vertex_end);
    auto bound = triqs_xca::block_sparse::bind::backbone(G, assignment, model.operators, left[i]);
    EXPECT_TRUE(bound.vertices.empty());
    ASSERT_EQ(bound.propagators.size(), 1);
    EXPECT_EQ(bound.propagator(3).data(), G.get_block(i).data());
    EXPECT_EQ(left[i].first_edge, (left[i].edge_end - 1));
    EXPECT_EQ(left[i].sector(left[i].first_vertex), i);
    EXPECT_EQ(left[i].sector(left[i].vertex_end), i);
  }
}

TEST(CorrelatorPlan, external_joins_prescribe_closure_and_observable_destinations) {
  Ladder model;
  auto G = model.propagator();
  auto assignment = triqs_xca::diagram::prepare_diagram(triqs_xca::diagram::Topology{nda::array<int, 2>{{0, 1}}}, 1,
                                       triqs_xca::diagram::Indices{{}, {}, {}}, nda::vector<double>{1.}, triqs_xca::diagram::Kind::correlator);
  auto right = fragments(G, assignment, model.operators, triqs_xca::diagram::CorrelatorSegment::zero_to_tau);
  auto left = fragments(G, assignment, model.operators, triqs_xca::diagram::CorrelatorSegment::tau_to_beta);
  ASSERT_EQ(right.size(), 3);
  ASSERT_EQ(left.size(), 3);
  std::vector<BlockOp> mu{model.raising}, kap{model.raising, model.lowering};
  std::vector<triqs_xca::block_sparse::diagram::CorrelatorDiagram> joins;
  triqs_xca::block_sparse::diagram::visit_correlator_diagrams(G, right, left, mu, kap, [&](auto const &j) { joins.push_back(j); });
  ASSERT_EQ(joins.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(joins[i].at_tau_operator, 0);
    EXPECT_EQ(joins[i].zero_to_tau_path, i);
    EXPECT_EQ(joins[i].tau_to_beta_path, i + 1);
    ASSERT_EQ(joins[i].at_zero_operators.size(), 1);
    EXPECT_EQ(joins[i].at_zero_operators[0], 1);
    auto operands = triqs_xca::block_sparse::bind::correlator(joins[i], right, left, mu, kap);
    EXPECT_EQ(operands.at_tau_matrix.data(), mu[0].get_block(i).data());
    EXPECT_EQ(operands.closures[0].matrix.data(), kap[1].get_block(i + 1).data());
  }
  auto masked = model.propagator({0, -1, 0});
  int count = 0;
  triqs_xca::block_sparse::diagram::visit_correlator_diagrams(masked, right, left, mu, kap, [&](auto const &) { ++count; });
  EXPECT_EQ(count, 0);
  left.erase(left.begin() + 1); // absent middle path; table indices differ from sectors
  joins.clear();
  triqs_xca::block_sparse::diagram::visit_correlator_diagrams(G, right, left, mu, kap, [&](auto const &j) { joins.push_back(j); });
  ASSERT_EQ(joins.size(), 1);
  EXPECT_EQ(joins[0].zero_to_tau_path, 1);
  EXPECT_EQ(joins[0].tau_to_beta_path, 1); // retained table index, not sector 2
  auto operands = triqs_xca::block_sparse::bind::correlator(joins[0], right, left, mu, kap);
  EXPECT_EQ(operands.at_tau_matrix.data(), mu[0].get_block(1).data());
  EXPECT_EQ(operands.closures[0].matrix.data(), kap[1].get_block(2).data());
}

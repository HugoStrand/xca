#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/correlator_plan.hpp>

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

std::vector<CorrelatorFragment> fragments(BlockDiagOpFun const &G, DiagramAssignment const &assignment,
                                         BlockOpSymQuartet const &operators, CorrelatorSide side) {
  std::vector<CorrelatorFragment> result;
  visit_correlator_fragments(G, assignment, operators, OperatorSupports(operators), side, [&](auto const &p) { result.push_back(p); });
  return result;
}
}

TEST(CorrelatorPlan, crossing_fragments_have_independent_owned_paths) {
  Ladder model;
  auto G = model.propagator();
  auto assignment = prepare_assignment(DiagramTopology{nda::array<int, 2>{{0, 2}, {1, 3}}}, 1,
                                       AssignmentIndices{{1}, {0}, {0}}, nda::vector<double>{1.}, DiagramKind::correlator);
  auto right = fragments(G, assignment, model.operators, CorrelatorSide::right);
  auto left = fragments(G, assignment, model.operators, CorrelatorSide::left);
  ASSERT_EQ(right.size(), 2);
  ASSERT_EQ(left.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(right[i].first_edge, 0);
    EXPECT_EQ(right[i].last_edge, 1);
    EXPECT_EQ(right[i].start_sector, i + 1);
    EXPECT_EQ(right[i].end_sector, i);
    EXPECT_EQ(right[i].propagator_blocks(0), i + 1);
    EXPECT_EQ(right[i].propagator_blocks(1), i);
    EXPECT_EQ(left[i].first_edge, 2);
    EXPECT_EQ(left[i].last_edge, 3);
    EXPECT_EQ(left[i].start_sector, i);
    EXPECT_EQ(left[i].end_sector, i + 1);
    EXPECT_EQ(left[i].propagator_blocks(2), i);
    EXPECT_EQ(left[i].propagator_blocks(3), i + 1);
  }
  auto bound = bind_correlator_fragment(G, assignment, model.operators, right[0]);
  EXPECT_EQ(bound.propagators[0].data(), G.get_block(1).data());
  EXPECT_EQ(bound.vertices[1].extent(0), 1);
  EXPECT_EQ(bound.vertices[1].extent(1), 2);
  right[0].dimensions = -99;
  EXPECT_EQ(bound.path.dimensions(1), 2);
  EXPECT_EQ(bound.path.dimensions(2), 1);

  auto masked = model.propagator({-1, 0, 0});
  auto surviving = fragments(masked, assignment, model.operators, CorrelatorSide::right);
  ASSERT_EQ(surviving.size(), 1);
  EXPECT_EQ(surviving[0].start_sector, 2);
  EXPECT_EQ(surviving[0].end_sector, 1);
}

TEST(CorrelatorPlan, empty_fragments_survive_a_failed_walk_on_the_other_side) {
  Ladder model;
  auto G = model.propagator();
  // The [tau,beta] side has no internal vertices. The preceding right walk
  // fails at its last starting sector; that must not poison the empty side.
  auto assignment = prepare_assignment(DiagramTopology{nda::array<int, 2>{{0, 3}, {1, 2}}}, 1,
                                       AssignmentIndices{{0}, {0}, {0}}, nda::vector<double>{1.}, DiagramKind::correlator);
  auto right = fragments(G, assignment, model.operators, CorrelatorSide::right);
  auto left = fragments(G, assignment, model.operators, CorrelatorSide::left);
  ASSERT_EQ(right.size(), 2);
  ASSERT_EQ(left.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(left[i].first_edge, left[i].last_edge);
    EXPECT_EQ(left[i].start_sector, i);
    EXPECT_EQ(left[i].end_sector, i);
  }
}

TEST(CorrelatorPlan, external_joins_prescribe_closure_and_observable_destinations) {
  Ladder model;
  auto G = model.propagator();
  auto assignment = prepare_assignment(DiagramTopology{nda::array<int, 2>{{0, 1}}}, 1,
                                       AssignmentIndices{{}, {}, {}}, nda::vector<double>{1.}, DiagramKind::correlator);
  auto right = fragments(G, assignment, model.operators, CorrelatorSide::right);
  auto left = fragments(G, assignment, model.operators, CorrelatorSide::left);
  ASSERT_EQ(right.size(), 3);
  ASSERT_EQ(left.size(), 3);
  nda::vector<int> right_ends{0, 1, 2}, left_ends{0, 1, 2};
  std::vector<BlockOp> mu{model.raising}, kap{model.raising, model.lowering};
  std::vector<CorrelatorJoin> joins;
  visit_correlator_joins(G, right_ends, left_ends, mu, kap, [&](auto const &j) { joins.push_back(j); });
  ASSERT_EQ(joins.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(joins[i].mu, 0);
    EXPECT_EQ(joins[i].right_start, i);
    EXPECT_EQ(joins[i].left_start, i + 1);
    ASSERT_EQ(joins[i].traces.size(), 1);
    EXPECT_EQ(joins[i].traces[0].kap, 1);
    EXPECT_EQ(joins[i].mu_op.data(), mu[0].get_block(i).data());
    EXPECT_EQ(joins[i].traces[0].op.data(), kap[1].get_block(i + 1).data());
  }
  auto masked = model.propagator({0, -1, 0});
  int count = 0;
  visit_correlator_joins(masked, right_ends, left_ends, mu, kap, [&](auto const &) { ++count; });
  EXPECT_EQ(count, 0);
  left_ends(1) = -1;
  joins.clear();
  visit_correlator_joins(G, right_ends, left_ends, mu, kap, [&](auto const &j) { joins.push_back(j); });
  ASSERT_EQ(joins.size(), 1);
  EXPECT_EQ(joins[0].right_start, 1);
}

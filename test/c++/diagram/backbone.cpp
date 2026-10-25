#include <gtest/gtest.h>
#include <triqs_xca/diagram/correlator.hpp>
#include <triqs_xca/diagram/diagram.hpp>

using triqs_xca::diagram::Direction;

namespace diagram = triqs_xca::diagram;

TEST(BackboneRange, correlator_geometry_is_shared) {
  diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  auto [zero_to_tau, tau_to_beta] = diagram::correlator_backbone_ranges(topology);
  EXPECT_EQ(zero_to_tau, (diagram::BackboneRange{1, 2, 0, 2}));
  EXPECT_EQ(tau_to_beta, (diagram::BackboneRange{3, 4, 2, 4}));
  diagram::BackboneRange full{0, 4, 0, 4};
  EXPECT_TRUE(full.contains(zero_to_tau));
  EXPECT_TRUE(full.contains(tau_to_beta));
  EXPECT_FALSE(zero_to_tau.contains(tau_to_beta));
}

TEST(BackboneRange, first_order_has_two_single_edge_segments) {
  diagram::Topology topology{nda::array<int, 2>{{0, 1}}};
  auto [zero_to_tau, tau_to_beta] = diagram::correlator_backbone_ranges(topology);
  EXPECT_EQ(zero_to_tau, (diagram::BackboneRange{1, 1, 0, 1}));
  EXPECT_EQ(tau_to_beta, (diagram::BackboneRange{2, 2, 1, 2}));
  EXPECT_EQ((diagram::BackboneRange{0, 2, 0, 1}).vertex_end, 2); // full self-energy includes outer vertices
}

TEST(BackboneRange, rejects_disconnected_or_reversed_intervals) {
  EXPECT_THROW((diagram::BackboneRange{0, 5, 0, 2}), std::invalid_argument);
  EXPECT_THROW((diagram::BackboneRange{0, 4, 2, 4}), std::invalid_argument);
  EXPECT_THROW((diagram::BackboneRange{3, 2, 2, 3}), std::invalid_argument);
  EXPECT_THROW((diagram::BackboneRange{1, 2, 1, 1}), std::invalid_argument); // vertex-only
  EXPECT_TRUE((diagram::BackboneRange{1, 1, 1, 1}).empty());
}


TEST(Backbone, full_extent_distinguishes_self_energy_and_correlator) {
  diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  auto sigma = diagram::prepare_diagram(topology, 1, diagram::Arcs{{Direction::forward, -1, -1}, {Direction::forward, 0, 0}}, nda::vector<double>{1.}, diagram::Kind::self_energy);
  auto corr = diagram::prepare_diagram(topology, 1, diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, 0, 0}}, nda::vector<double>{1.}, diagram::Kind::correlator);
  EXPECT_EQ(sigma.backbone.range(), (diagram::BackboneRange{0, 4, 0, 3}));
  EXPECT_EQ(corr.backbone.range(), (diagram::BackboneRange{0, 4, 0, 4}));
  auto [zero_to_tau, tau_to_beta] = diagram::correlator_backbone_ranges(topology);
  EXPECT_TRUE(corr.backbone.range().contains(tau_to_beta));
  EXPECT_FALSE(sigma.backbone.range().contains(tau_to_beta));
  // Internal arc (1,3) occupies edges 1 and 2, never the edge from vertex 3 to beta.
  EXPECT_EQ(corr.backbone.edges[1].kernels, (std::vector<diagram::KernelFactor>{{1, 1}}));
  EXPECT_EQ(corr.backbone.edges[2].kernels, (std::vector<diagram::KernelFactor>{{1, 1}}));
  EXPECT_TRUE(corr.backbone.edges[3].kernels.empty());
  auto copy = corr.backbone;
  copy.vertices[1].arc_idx = 0;
  copy.edges[1].kernels.clear();
  EXPECT_EQ(corr.arcs[corr.backbone.vertices[1].arc_idx].retained_flavour, 0);
  EXPECT_EQ(corr.backbone.edges[1].kernels, (std::vector<diagram::KernelFactor>{{1, 1}}));
}

TEST(DiagramEndpoints, ranges_and_sum_distinguish_the_two_kinds) {
  for (auto topology_arr : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 3}, {1, 2}}}) {
    diagram::Topology topology{topology_arr};
    int end = 2 * topology.order(), special = end - 1;
    for (long direction : {0, 1}) {
      // One flavour and pole makes the least significant digit the special direction.
      auto sigma = diagram::prepare_diagram(topology, 1, nda::vector<double>{1.}, direction, diagram::Kind::self_energy);
      EXPECT_EQ(sigma.kind, diagram::Kind::self_energy);
      auto ranges = sigma.endpoint_ranges();
      EXPECT_EQ(ranges.before_special, (diagram::BackboneRange{1, special, 0, special}));
      EXPECT_EQ(ranges.after_special, (diagram::BackboneRange{end, end, special, special}));
      EXPECT_TRUE(ranges.after_special.empty());
      EXPECT_TRUE(sigma.backbone.range().contains(ranges.after_special));
      auto sum = sigma.self_energy_endpoint_sum(3);
      EXPECT_EQ(sum.flavours, 3);
      EXPECT_EQ(sum.orientation.forward, direction == 1);
    }
    auto corr = diagram::prepare_diagram(topology, 1, nda::vector<double>{1.}, 0, diagram::Kind::correlator);
    EXPECT_EQ(corr.kind, diagram::Kind::correlator);
    auto ranges = corr.endpoint_ranges();
    EXPECT_EQ(ranges.after_special, (diagram::BackboneRange{end, end, special, end}));
    EXPECT_FALSE(ranges.after_special.empty()); // the final propagator remains
    auto legacy = diagram::correlator_backbone_ranges(topology);
    EXPECT_EQ(ranges.before_special, legacy.first);
    EXPECT_EQ(ranges.after_special, legacy.second);
    EXPECT_THROW(corr.self_energy_endpoint_sum(3), std::invalid_argument);
  }
  auto sigma = diagram::prepare_diagram(diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}}, 1,
                                       nda::vector<double>{1.}, 0, diagram::Kind::self_energy);
  EXPECT_EQ(sigma.endpoint_ranges().after_special, (diagram::BackboneRange{3, 4, 2, 3}));
  EXPECT_FALSE(sigma.endpoint_ranges().after_special.empty());
  EXPECT_THROW(sigma.self_energy_endpoint_sum(-1), std::invalid_argument);
}

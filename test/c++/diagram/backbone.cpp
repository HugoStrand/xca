#include <gtest/gtest.h>
#include <triqs_xca/diagram/backbone.hpp>

using triqs_xca::diagram::Direction;

namespace diagram = triqs_xca::diagram;

TEST(BackboneRange, correlator_geometry_is_shared) {
  diagram::Topology topology{nda::array<int, 2>{{0, 2}, {1, 3}}};
  std::vector<triqs_xca::Statistics> statistics{triqs_xca::Statistics::fermionic};
  auto arcs = diagram::prepare_arcs(diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, 0, 0}}, statistics,
                                   nda::vector<double>{1.}, diagram::Kind::correlator);
  auto backbone = diagram::prepare_backbone(topology, arcs);
  auto [zero_to_tau, tau_to_beta] = backbone.endpoint_ranges();
  EXPECT_EQ(zero_to_tau, (diagram::BackboneRange{1, 2, 0, 2}));
  EXPECT_EQ(tau_to_beta, (diagram::BackboneRange{3, 4, 2, 4}));
  diagram::BackboneRange full{0, 4, 0, 4};
  EXPECT_TRUE(full.contains(zero_to_tau));
  EXPECT_TRUE(full.contains(tau_to_beta));
  EXPECT_FALSE(zero_to_tau.contains(tau_to_beta));
}

TEST(BackboneRange, first_order_has_two_single_edge_segments) {
  diagram::Topology topology{nda::array<int, 2>{{0, 1}}};
  auto backbone = diagram::prepare_backbone(topology, diagram::Arcs{{Direction::external, -1, -1}});
  auto [zero_to_tau, tau_to_beta] = backbone.endpoint_ranges();
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
  std::vector<triqs_xca::Statistics> statistics{triqs_xca::Statistics::fermionic};
  auto sigma = diagram::prepare_backbone(topology,
                                         diagram::prepare_arcs(diagram::Arcs{{Direction::forward, -1, -1}, {Direction::forward, 0, 0}}, statistics,
                                                               nda::vector<double>{1.}, diagram::Kind::self_energy));
  auto corr  = diagram::prepare_backbone(topology,
                                         diagram::prepare_arcs(diagram::Arcs{{Direction::external, -1, -1}, {Direction::forward, 0, 0}}, statistics,
                                                               nda::vector<double>{1.}, diagram::Kind::correlator));
  EXPECT_EQ(sigma.range(), (diagram::BackboneRange{0, 4, 0, 3}));
  EXPECT_EQ(corr.range(), (diagram::BackboneRange{0, 4, 0, 4}));
  auto [zero_to_tau, tau_to_beta] = corr.endpoint_ranges();
  EXPECT_TRUE(corr.range().contains(tau_to_beta));
  EXPECT_FALSE(sigma.range().contains(tau_to_beta));
  // Internal arc (1,3) occupies edges 1 and 2, never the edge from vertex 3 to beta.
  EXPECT_EQ(corr.edges[1].kernels, (std::vector<diagram::KernelFactor>{{0, 1}}));
  EXPECT_EQ(corr.edges[2].kernels, (std::vector<diagram::KernelFactor>{{0, 1}}));
  EXPECT_TRUE(corr.edges[3].kernels.empty());
  auto copy                = corr;
  copy.vertices[1].flavour = -1;
  copy.edges[1].kernels.clear();
  EXPECT_EQ(corr.vertices[1].flavour, 0);
  EXPECT_EQ(corr.edges[1].kernels, (std::vector<diagram::KernelFactor>{{0, 1}}));
}

TEST(BackboneEndpoints, ranges_and_operations_distinguish_the_two_kinds) {
  std::vector<triqs_xca::Statistics> statistics{triqs_xca::Statistics::fermionic};
  for (auto topology_arr : {nda::array<int, 2>{{0, 1}}, nda::array<int, 2>{{0, 3}, {1, 2}}}) {
    diagram::Topology topology{topology_arr};
    int end = 2 * topology.order(), special = end - 1;
    for (long direction : {0, 1}) {
      // One flavour and pole makes the least significant digit the special direction.
      auto sigma = diagram::prepare_backbone(
         topology, diagram::decode_arcs(topology.order(), statistics, nda::vector<double>{1.}, direction, diagram::Kind::self_energy));
      EXPECT_EQ(static_cast<int>(sigma.endpoints.direction), direction);
      EXPECT_EQ(sigma.endpoints.special_vertex, special);
      EXPECT_NO_THROW(sigma.require_self_energy());
      EXPECT_THROW(sigma.require_correlator(), std::invalid_argument);
      auto ranges = sigma.endpoint_ranges();
      EXPECT_EQ(ranges.before_special, (diagram::BackboneRange{1, special, 0, special}));
      EXPECT_EQ(ranges.after_special, (diagram::BackboneRange{end, end, special, special}));
      EXPECT_TRUE(ranges.after_special.empty());
      EXPECT_TRUE(sigma.range().contains(ranges.after_special));
      // All internal directions are backward for these small flat indices.
      // Their sign cancels the diagram-order sign in both tested orders.
      EXPECT_EQ(sigma.self_energy_sign(), direction == 0 ? -1 : 1);
    }
    auto corr =
       diagram::prepare_backbone(topology, diagram::decode_arcs(topology.order(), statistics, nda::vector<double>{1.}, 0, diagram::Kind::correlator));
    EXPECT_EQ(corr.endpoints.direction, Direction::external);
    EXPECT_NO_THROW(corr.require_correlator());
    EXPECT_THROW(corr.require_self_energy(), std::invalid_argument);
    EXPECT_THROW(corr.self_energy_sign(), std::invalid_argument);
    auto ranges = corr.endpoint_ranges();
    EXPECT_EQ(ranges.before_special, (diagram::BackboneRange{1, special, 0, special}));
    EXPECT_EQ(ranges.after_special, (diagram::BackboneRange{end, end, special, end}));
    EXPECT_FALSE(ranges.after_special.empty()); // the final propagator remains
  }
  auto sigma = diagram::prepare_backbone(diagram::Topology{nda::array<int, 2>{{0, 2}, {1, 3}}},
                                         diagram::decode_arcs(2, statistics, nda::vector<double>{1.}, 0, diagram::Kind::self_energy));
  EXPECT_EQ(sigma.endpoint_ranges().after_special, (diagram::BackboneRange{3, 4, 2, 3}));
  EXPECT_FALSE(sigma.endpoint_ranges().after_special.empty());
}

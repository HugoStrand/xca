#include <gtest/gtest.h>
#include <triqs_xca/diagram/correlator.hpp>

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
  EXPECT_THROW((diagram::BackboneRange{1, 1, 1, 1}), std::invalid_argument);
}

#pragma once
#include "arc_catalog.hpp"
#include "backbone_range.hpp"
#include "topology.hpp"
#include <utility>

namespace triqs_xca::diagram {

  /// Ranges excluding the external vertices: first zero-to-tau, then tau-to-beta.
  /// This is diagram geometry, independent of traversal or operator representation.
  std::pair<BackboneRange, BackboneRange> correlator_backbone_ranges(Topology const &topology);

  /// External operator lists have homogeneous statistics at each endpoint. These
  /// are observable metadata, not an interaction flavour or sentinel orbital.
  struct CorrelatorStatistics {
    Statistics at_zero = Statistics::fermionic;
    Statistics at_tau  = Statistics::fermionic;
  };

  /// Compatibility adapter for the existing uniform-statistics correlator API.
  CorrelatorStatistics uniform_correlator_statistics(bool fermionic);

} // namespace triqs_xca::diagram

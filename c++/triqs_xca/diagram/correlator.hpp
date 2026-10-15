#pragma once
#include "arc_catalog.hpp"

namespace triqs_xca::diagram {

/// External operator lists have homogeneous statistics at each endpoint. These
/// are observable metadata, not an interaction flavour or sentinel orbital.
struct CorrelatorStatistics {
  Statistics at_zero = Statistics::fermionic;
  Statistics at_tau = Statistics::fermionic;
};

/// Compatibility adapter for the existing uniform-statistics correlator API.
CorrelatorStatistics uniform_correlator_statistics(bool fermionic);

} // namespace triqs_xca::diagram

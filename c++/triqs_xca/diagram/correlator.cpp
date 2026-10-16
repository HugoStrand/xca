#include "correlator.hpp"

namespace triqs_xca::diagram {
  std::pair<BackboneRange, BackboneRange> correlator_backbone_ranges(Topology const &topology) {
    int special = topology.special_vertex(), end = 2 * topology.order();
    return {{1, special, 0, special}, {special + 1, end, special, end}};
  }


  CorrelatorStatistics uniform_correlator_statistics(bool fermionic) {
    auto statistics = fermionic ? Statistics::fermionic : Statistics::bosonic;
    return {statistics, statistics};
  }

} // namespace triqs_xca::diagram

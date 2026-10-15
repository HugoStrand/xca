#include "correlator.hpp"

namespace triqs_xca::diagram {

CorrelatorStatistics uniform_correlator_statistics(bool fermionic) {
  auto statistics = fermionic ? Statistics::fermionic : Statistics::bosonic;
  return {statistics, statistics};
}

} // namespace triqs_xca::diagram

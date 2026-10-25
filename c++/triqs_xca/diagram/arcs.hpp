#pragma once
#include "arc.hpp"
#include <vector>

namespace triqs_xca::diagram {
  enum class Kind { self_energy, correlator };

  /// Arc choices indexed exactly like Topology::arcs, including the special row zero.
  using Arcs = std::vector<Arc>;

  void validate_arcs(Arcs const &arcs, int order, Kind kind);
  long count_diagrams(int order, int flavours, long pole_count, Kind kind);
  Arcs decode_arcs(int order, int flavours, long pole_count, long flat_index, Kind kind);
} // namespace triqs_xca::diagram

#pragma once
#include "arc.hpp"
#include <vector>

namespace triqs_xca::diagram {
  
  enum class Direction { external = -1, backward = 0, forward = 1 };

  /// One arc choice. Arc zero has no retained flavour or pole; for a correlator
  /// it denotes the external operator pair rather than an interaction.
  struct Arc {
    Direction direction                = Direction::external;
    int retained_flavour               = -1;
    int pole_index                     = -1;
    bool operator==(Arc const &) const = default;
  };

  enum class Kind { self_energy, correlator };

  /// Arc choices indexed exactly like Topology::arcs, including the special row zero.
  using Arcs = std::vector<Arc>;

  void validate_arcs(Arcs const &arcs, int order, Kind kind);
  long count_diagrams(int order, int flavours, long pole_count, Kind kind);
  Arcs decode_arcs(int order, int flavours, long pole_count, long flat_index, Kind kind);
} // namespace triqs_xca::diagram

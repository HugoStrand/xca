#pragma once

#include <vector>
#include <iosfwd>

#include <nda/nda.hpp>
#include "topology.hpp"
#include "arcs.hpp"
#include "backbone.hpp"

namespace triqs_xca::diagram {

  /// Owned description of a diagram and its stable scalar-kernel recipe.
  /// Prepared recipes derive from topology and arcs: prepare again after changing
  /// either input or the pole signs. Consumers take a const reference; copying this value shares no mutable arrays.
  /// No sector indices, matrix storage, or executor workspace belong here.
  struct Diagram {
    Topology topology;
    Arcs arcs;
    Backbone backbone;
    TauKernelPrefactor kernel_prefactor;
    int direction_sign;
  };

  std::ostream &operator<<(std::ostream &out, Diagram const &diagram);

  Diagram prepare_diagram(Topology const &topology, int flavours, Arcs const &arcs, nda::vector_const_view<double> poles, Kind kind);
  Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles, long flat_index, Kind kind);

  // Shared logical preparation of vertex and kernel recipes.
  // They return fresh values: no previous diagram or reset is required.
  std::vector<Vertex> prepare_vertices(Topology const &topology, Arcs const &arcs, Kind kind);

} // namespace triqs_xca::diagram

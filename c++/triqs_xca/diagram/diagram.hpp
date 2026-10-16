#pragma once

#include <vector>
#include <iosfwd>

#include <nda/nda.hpp>
#include "topology.hpp"
#include "indices.hpp"
#include "backbone.hpp"

namespace triqs_xca::diagram {

  /// Owned description of a diagram and its stable scalar-kernel recipe.
  /// Consumers take a const reference; copying this value shares no mutable arrays.
  /// No sector indices, matrix storage, or executor workspace belong here.
  struct Diagram {
    Topology topology;
    Backbone backbone;
    nda::vector<int> directions;   // per arc; -1 for the absent external correlator arc
    nda::vector<int> pole_indices; // internal arcs only
    TauKernelPrefactor kernel_prefactor;
    int direction_sign;
  };

  std::ostream &operator<<(std::ostream &out, Diagram const &diagram);

  Diagram prepare_diagram(Topology const &topology, int flavours, Indices const &indices, nda::vector_const_view<double> poles, Kind kind);
  Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles, long flat_index, Kind kind);

  // Shared logical preparation of vertex and kernel recipes.
  // They return fresh values: no previous diagram or reset is required.
  std::vector<Vertex> prepare_vertices(Topology const &topology, nda::vector_const_view<int> directions, Kind kind);

} // namespace triqs_xca::diagram

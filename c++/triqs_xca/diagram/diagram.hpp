#pragma once

#include <vector>
#include <iosfwd>

#include <nda/nda.hpp>
#include "topology.hpp"
#include "indices.hpp"
#include "tau_kernel.hpp"

namespace triqs_xca::diagram {

/// One vertex in an assigned diagram. An orbital of -1 is an unresolved special
/// endpoint, to be supplied by the remaining endpoint sum or observable.
struct Vertex {
  bool barred;
  bool adjoint;
  int flavour;
  int pole_slot;
};

/// Owned description of an assignment and its stable scalar-kernel recipe.
/// Consumers take a const reference; copying this value shares no mutable arrays.
/// No sector indices, matrix storage, or executor workspace belong here.
struct Diagram {
  Topology topology;
  std::vector<Vertex> vertices;
  nda::vector<int> directions; // per arc; -1 for the absent external correlator arc
  nda::vector<int> pole_indices;      // internal arcs only
  TauKernelPlacement kernel_placement;
  int direction_sign;
};

std::ostream &operator<<(std::ostream &out, Diagram const &assignment);

Diagram prepare_diagram(Topology const &topology, int flavours, Indices const &indices,
                                     nda::vector_const_view<double> poles, Kind kind);
Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles,
                                     long flat_index, Kind kind);

// Shared logical preparation of vertex and kernel recipes.
// They return fresh values: no previous assignment or reset is required.
std::vector<Vertex> prepare_vertices(Topology const &topology, nda::vector_const_view<int> directions, Kind kind);

} // namespace triqs_xca::diagram

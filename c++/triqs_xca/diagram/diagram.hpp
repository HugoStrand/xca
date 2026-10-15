#pragma once

#include <vector>
#include <iosfwd>

#include <nda/nda.hpp>
#include "topology.hpp"
#include "indices.hpp"

namespace triqs_xca::diagram {

/// One vertex in an assigned diagram. An orbital of -1 is an unresolved special
/// endpoint, to be supplied by the remaining endpoint sum or observable.
struct Vertex {
  bool barred;
  bool dagger;
  int orbital;
  int pole_slot;
  int kernel_sign;
};

/// Owned description of an assignment and its stable scalar-kernel recipe.
/// Consumers take a const reference; copying this value shares no mutable arrays.
/// No sector indices, matrix storage, or executor workspace belong here.
struct Diagram {
  Topology topology;
  long flat_index; // legacy selection identity, not an execution schedule
  std::vector<Vertex> vertices;
  nda::vector<int> directions; // per arc; -1 for the absent external correlator arc
  nda::vector<int> poles;      // internal arcs only
  nda::array<int, 2> edge_kernels;
  nda::vector<int> prefactor_kernel_signs;
  nda::vector<int> prefactor_kernel_exponents;
  int prefactor_sign;
};

std::ostream &operator<<(std::ostream &out, Diagram const &assignment);

Diagram prepare_diagram(Topology const &topology, int flavours, Indices const &indices,
                                     nda::vector_const_view<double> poles, Kind kind);
Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles,
                                     long flat_index, Kind kind);

// Shared logical preparation of vertex and kernel recipes.
// They return fresh values: no previous assignment or reset is required.
std::vector<Vertex> prepare_vertices(Topology const &topology, nda::vector_const_view<int> directions, Kind kind);

struct VertexKernel {
  int pole_slot = 0;
  int sign = 0;
};
struct KernelFactors {
  std::vector<VertexKernel> vertices;
  nda::array<int, 2> edges;
  nda::vector<int> signs;
  nda::vector<int> exponents;
  int sign = 1;
};
KernelFactors prepare_kernel_factors(Topology const &topology, nda::vector_const_view<int> full_directions,
                                    nda::vector_const_view<int> pole_indices, nda::vector_const_view<double> poles);

} // namespace triqs_xca::diagram

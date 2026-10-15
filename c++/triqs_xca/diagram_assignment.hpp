#pragma once

#include <vector>

#include <nda/nda.hpp>

namespace triqs_xca::backbone {
class Backbone;
}

namespace triqs_xca {

/// Diagram connectivity, independent of operator representation and sector layout.
struct DiagramTopology {
  nda::array<int, 2> arcs;

  int order() const { return static_cast<int>(arcs.extent(0)); }
  int special_vertex() const { return arcs(0, 1); }
};

/// One vertex in an assigned diagram. An orbital of -1 is an unresolved special
/// endpoint, to be supplied by the remaining endpoint sum or observable.
struct VertexAssignment {
  bool barred;
  bool dagger;
  int orbital;
  int pole_slot;
  int kernel_sign;
};

/// Owned description of an assignment and its stable scalar-kernel recipe.
/// Consumers take a const reference; copying this value shares no mutable arrays.
/// No sector indices, matrix storage, or executor workspace belong here.
struct DiagramAssignment {
  DiagramTopology topology;
  long flat_index; // legacy selection identity, not an execution schedule
  std::vector<VertexAssignment> vertices;
  nda::vector<int> directions; // per arc; -1 for the absent external correlator arc
  nda::vector<int> poles;      // internal arcs only
  nda::array<int, 2> edge_kernels;
  nda::vector<int> prefactor_kernel_signs;
  nda::vector<int> prefactor_kernel_exponents;
  int prefactor_sign;
};

/// Temporary migration bridge: capture a fully assigned legacy backbone.
/// The two special endpoint orbitals remain unresolved even if an earlier
/// numerical contraction has assigned them in the mutable backbone. This bridge
/// will be replaced by preparation directly from topology and logical indices.
DiagramAssignment snapshot_assignment(backbone::Backbone &source);

} // namespace triqs_xca

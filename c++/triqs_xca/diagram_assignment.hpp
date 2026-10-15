#pragma once

#include <vector>
#include <iosfwd>

#include <nda/nda.hpp>

namespace triqs_xca {

/// Diagram connectivity, independent of operator representation and sector layout.
struct DiagramTopology {
  nda::array<int, 2> arcs;

  explicit DiagramTopology(nda::array<int, 2> arcs);

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

std::ostream &operator<<(std::ostream &out, DiagramAssignment const &assignment);

enum class DiagramKind { self_energy, correlator };

/// Logical indices: one direction per interaction arc (the external correlator
/// arc is absent), one pole and one orbital per internal arc. No linear index
/// or execution order is needed to prepare these assignments.
struct AssignmentIndices {
  nda::vector<int> directions;
  nda::vector<int> poles;
  nda::vector<int> orbitals;
};

/// The legacy flat-index convention lives at the selection boundary.
long count_assignments(int order, int flavours, long pole_count, DiagramKind kind);
AssignmentIndices decode_assignment_indices(int order, int flavours, long pole_count, long flat_index, DiagramKind kind);
DiagramAssignment prepare_assignment(DiagramTopology const &topology, int flavours, AssignmentIndices const &indices,
                                     nda::vector_const_view<double> poles, DiagramKind kind);
DiagramAssignment prepare_assignment(DiagramTopology const &topology, int flavours, nda::vector_const_view<double> poles,
                                     long flat_index, DiagramKind kind);

// Shared logical preparation of vertex and kernel recipes.
// They return fresh values: no previous assignment or reset is required.
std::vector<VertexAssignment> prepare_vertices(DiagramTopology const &topology, nda::vector_const_view<int> directions, DiagramKind kind);

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
KernelFactors prepare_kernel_factors(DiagramTopology const &topology, nda::vector_const_view<int> full_directions,
                                    nda::vector_const_view<int> pole_indices, nda::vector_const_view<double> poles);

} // namespace triqs_xca

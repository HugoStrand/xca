#include "diagram_assignment.hpp"
#include "interaction.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

namespace triqs_xca {
namespace {
long checked_power(long base, int exponent) {
  long value = 1;
  for (int i = 0; i < exponent; ++i)
    if (__builtin_mul_overflow(value, base, &value)) throw std::overflow_error("assignment count does not fit in a long");
  return value;
}
int direction_count(int order, DiagramKind kind) { return order - (kind == DiagramKind::correlator); }
}

DiagramTopology::DiagramTopology(nda::array<int, 2> input) : arcs(std::move(input)) {
  int m = order();
  if (m < 1 || arcs.extent(1) != 2) throw std::invalid_argument("topology must contain at least one pair of vertices");
  std::vector<int> vertices(2 * m);
  for (int a = 0; a < m; ++a) {
    if (a > 0 && arcs(a, 0) <= arcs(a - 1, 0)) throw std::invalid_argument("first pair indices must be in increasing order");
    if (arcs(a, 0) >= arcs(a, 1)) throw std::invalid_argument("first row of topology must contain smaller-numbered vertices");
    vertices[2 * a] = arcs(a, 0);
    vertices[2 * a + 1] = arcs(a, 1);
  }
  std::ranges::sort(vertices);
  for (int v = 0; v < 2 * m; ++v)
    if (vertices[v] != v) throw std::invalid_argument("topology must contain each vertex index from 0 to 2m-1 exactly once");
}

long count_assignments(int order, int flavours, long pole_count, DiagramKind kind) {
  if (order < 1 || flavours < 0 || pole_count < 0) throw std::invalid_argument("invalid assignment space dimensions");
  long count = checked_power(2, direction_count(order, kind));
  long orbitals = checked_power(flavours, order - 1);
  if (__builtin_mul_overflow(count, orbitals, &count)) throw std::overflow_error("assignment count does not fit in a long");
  for (int p = 0; p < order - 1; ++p)
    if (__builtin_mul_overflow(count, pole_count, &count)) throw std::overflow_error("assignment count does not fit in a long");
  return count;
}

AssignmentIndices decode_assignment_indices(int order, int flavours, long pole_count, long flat_index, DiagramKind kind) {
  long count = count_assignments(order, flavours, pole_count, kind);
  if (flat_index < 0 || flat_index >= count)
    throw std::invalid_argument("flat index " + std::to_string(flat_index) + " is out of range [0, " + std::to_string(count) + ")");
  AssignmentIndices result{nda::vector<int>(direction_count(order, kind)), nda::vector<int>(order - 1), nda::vector<int>(order - 1)};
  long remaining = flat_index;
  for (int a = 0; a < order - 1; ++a) {
    result.orbitals(a) = remaining % flavours;
    remaining /= flavours;
  }
  for (int a = 0; a < order - 1; ++a) {
    result.poles(a) = remaining % pole_count;
    remaining /= pole_count;
  }
  for (int a = 0; a < result.directions.size(); ++a) {
    result.directions(a) = remaining % 2;
    remaining /= 2;
  }
  return result;
}

std::vector<VertexAssignment> prepare_vertices(DiagramTopology const &topology, nda::vector_const_view<int> directions, DiagramKind kind) {
  int m = topology.order();
  int first = kind == DiagramKind::correlator ? 1 : 0;
  if (directions.size() != m - first) throw std::invalid_argument("wrong number of arc directions");
  std::vector<VertexAssignment> result(2 * m, {false, false, -1, 0, 0});
  // Legacy external correlator endpoint convention; statistics are assigned separately.
  result[0].dagger = true;
  for (int a = first; a < m; ++a) {
    int direction = directions(a - first);
    if (direction != 0 && direction != 1) throw std::invalid_argument("arc direction must be zero or one");
    InteractionLine line{direction == 1};
    result[topology.arcs(a, 0)].dagger = line.early_operator(-1).adjoint;
    result[topology.arcs(a, 1)].dagger = line.late_operator(-1).adjoint;
    result[topology.arcs(a, 1)].barred = a != 0;
  }
  return result;
}

KernelFactors prepare_kernel_factors(DiagramTopology const &topology, nda::vector_const_view<int> full_directions,
                                    nda::vector_const_view<int> pole_indices, nda::vector_const_view<double> poles) {
  int m = topology.order();
  if (full_directions.size() != m || pole_indices.size() != m - 1) throw std::invalid_argument("wrong number of kernel indices");
  KernelFactors result{std::vector<VertexKernel>(2 * m), nda::zeros<int>(2 * m - 1, m - 1), nda::vector<int>(m - 1), nda::vector<int>(m - 1)};
  for (int a = 1; a < m; ++a) {
    int p = a - 1, pole = pole_indices(p);
    if (pole < 0 || pole >= poles.size()) throw std::invalid_argument("pole index out of range");
    int direction = full_directions(a);
    if (direction != 0 && direction != 1) throw std::invalid_argument("internal arc direction must be zero or one");
    bool forward = direction == 1;
    int right = topology.arcs(a, 0), left = topology.arcs(a, 1);
    result.vertices[left].pole_slot = p;
    if (!forward) result.sign *= -1;
    // Avoid exponentially growing kernels: select vertex or edge placement by
    // the direction and pole sign. Zero poles belong to the vertex branch.
    if (forward ? poles(pole) <= 0 : poles(pole) >= 0) {
      int sign = forward ? -1 : 1;
      result.vertices[right] = {p, sign};
      result.vertices[left].sign = -sign;
      result.signs(p) = sign;
      result.exponents(p) = 1;
    } else {
      int sign = forward ? 1 : -1;
      for (int e = right; e < left; ++e) result.edges(e, p) = sign;
      result.signs(p) = sign;
      result.exponents(p) = left - right - 1;
    }
  }
  return result;
}

DiagramAssignment prepare_assignment(DiagramTopology const &topology, int flavours, AssignmentIndices const &indices,
                                     nda::vector_const_view<double> poles, DiagramKind kind) {
  int m = topology.order();
  if (flavours < 0 || indices.orbitals.size() != m - 1) throw std::invalid_argument("invalid orbital assignment dimensions");
  auto vertices = prepare_vertices(topology, indices.directions, kind);
  nda::vector<int> full_directions(m);
  int first = kind == DiagramKind::correlator ? 1 : 0;
  full_directions(0) = -1;
  full_directions(nda::range(first, m)) = indices.directions;
  auto kernels = prepare_kernel_factors(topology, full_directions, indices.poles, poles);
  for (int a = 1; a < m; ++a) {
    int orbital = indices.orbitals(a - 1);
    if (orbital < 0 || orbital >= flavours) throw std::invalid_argument("orbital index out of range");
    vertices[topology.arcs(a, 0)].orbital = orbital;
    vertices[topology.arcs(a, 1)].orbital = orbital;
  }
  for (int v = 0; v < 2 * m; ++v) {
    vertices[v].pole_slot = kernels.vertices[v].pole_slot;
    vertices[v].kernel_sign = kernels.vertices[v].sign;
  }
  return {topology, -1, std::move(vertices), std::move(full_directions), indices.poles, std::move(kernels.edges),
          std::move(kernels.signs), std::move(kernels.exponents), kernels.sign};
}

DiagramAssignment prepare_assignment(DiagramTopology const &topology, int flavours, nda::vector_const_view<double> poles,
                                     long flat_index, DiagramKind kind) {
  auto indices = decode_assignment_indices(topology.order(), flavours, poles.size(), flat_index, kind);
  auto result = prepare_assignment(topology, flavours, indices, poles, kind);
  result.flat_index = flat_index;
  return result;
}

std::ostream &operator<<(std::ostream &out, DiagramAssignment const &assignment) {
  out << "Topology:\n" << assignment.topology.arcs << "\nFlat index: " << assignment.flat_index
      << "\nDirections: " << assignment.directions << "\nPoles: " << assignment.poles << "\nVertices (orbital, dagger, bar, pole slot, kernel sign):\n";
  for (int v = 0; v < assignment.vertices.size(); ++v) {
    auto const &vertex = assignment.vertices[v];
    out << v << ": " << vertex.orbital << ' ' << vertex.dagger << ' ' << vertex.barred << ' ' << vertex.pole_slot << ' ' << vertex.kernel_sign << '\n';
  }
  return out << "Edge kernels:\n" << assignment.edge_kernels << "\nPrefactor: " << assignment.prefactor_sign
             << ", kernel signs " << assignment.prefactor_kernel_signs << ", exponents " << assignment.prefactor_kernel_exponents << '\n';
}

} // namespace triqs_xca

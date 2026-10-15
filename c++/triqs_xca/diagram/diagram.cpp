#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/diagram/arc.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

namespace triqs_xca::diagram {
std::vector<Vertex> prepare_vertices(Topology const &topology, nda::vector_const_view<int> directions, Kind kind) {
  int m = topology.order();
  int first = kind == Kind::correlator ? 1 : 0;
  if (directions.size() != m - first) throw std::invalid_argument("wrong number of arc directions");
  std::vector<Vertex> result(2 * m, {false, false, -1, 0, 0});
  // Legacy external correlator endpoint convention; statistics are assigned separately.
  result[0].dagger = true;
  for (int a = first; a < m; ++a) {
    int direction = directions(a - first);
    if (direction != 0 && direction != 1) throw std::invalid_argument("arc direction must be zero or one");
    ArcOrientation line{direction == 1};
    result[topology.arcs(a, 0)].dagger = line.early_operator(-1).adjoint;
    result[topology.arcs(a, 1)].dagger = line.late_operator(-1).adjoint;
    result[topology.arcs(a, 1)].barred = a != 0;
  }
  return result;
}

KernelFactors prepare_kernel_factors(Topology const &topology, nda::vector_const_view<int> full_directions,
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

Diagram prepare_diagram(Topology const &topology, int flavours, Indices const &indices,
                                     nda::vector_const_view<double> poles, Kind kind) {
  int m = topology.order();
  if (flavours < 0 || indices.orbitals.size() != m - 1) throw std::invalid_argument("invalid orbital assignment dimensions");
  auto vertices = prepare_vertices(topology, indices.directions, kind);
  nda::vector<int> full_directions(m);
  int first = kind == Kind::correlator ? 1 : 0;
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

Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles,
                                     long flat_index, Kind kind) {
  auto indices = decode_indices(topology.order(), flavours, poles.size(), flat_index, kind);
  auto result = prepare_diagram(topology, flavours, indices, poles, kind);
  result.flat_index = flat_index;
  return result;
}

std::ostream &operator<<(std::ostream &out, Diagram const &assignment) {
  out << "Topology:\n" << assignment.topology.arcs << "\nFlat index: " << assignment.flat_index
      << "\nDirections: " << assignment.directions << "\nPoles: " << assignment.poles << "\nVertices (orbital, dagger, bar, pole slot, kernel sign):\n";
  for (int v = 0; v < assignment.vertices.size(); ++v) {
    auto const &vertex = assignment.vertices[v];
    out << v << ": " << vertex.orbital << ' ' << vertex.dagger << ' ' << vertex.barred << ' ' << vertex.pole_slot << ' ' << vertex.kernel_sign << '\n';
  }
  return out << "Edge kernels:\n" << assignment.edge_kernels << "\nPrefactor: " << assignment.prefactor_sign
             << ", kernel signs " << assignment.prefactor_kernel_signs << ", exponents " << assignment.prefactor_kernel_exponents << '\n';
}

} // namespace triqs_xca::diagram

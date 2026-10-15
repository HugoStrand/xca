#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/diagram/arc.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

namespace triqs_xca::diagram {
  std::vector<Vertex> prepare_vertices(Topology const &topology, nda::vector_const_view<int> directions, Kind kind) {
    int m     = topology.order();
    int first = kind == Kind::correlator ? 1 : 0;
    if (directions.size() != m - first) throw std::invalid_argument("wrong number of arc directions");
    std::vector<Vertex> result(2 * m, {false, false, -1, 0});
    // Legacy external correlator endpoint convention; statistics are assigned separately.
    result[0].adjoint = true;
    for (int a = first; a < m; ++a) {
      int direction = directions(a - first);
      if (direction != 0 && direction != 1) throw std::invalid_argument("arc direction must be zero or one");
      ArcOrientation orientation{direction == 1};
      result[topology.arcs(a, 0)].adjoint = orientation.early_operator(-1).adjoint;
      result[topology.arcs(a, 1)].adjoint = orientation.late_operator(-1).adjoint;
      result[topology.arcs(a, 1)].barred  = a != 0;
    }
    return result;
  }

  Diagram prepare_diagram(Topology const &topology, int flavours, Indices const &indices, nda::vector_const_view<double> poles, Kind kind) {
    int m = topology.order();
    if (flavours < 0 || indices.flavours.size() != m - 1) throw std::invalid_argument("invalid orbital assignment dimensions");
    auto vertices = prepare_vertices(topology, indices.directions, kind);
    nda::vector<int> full_directions(m);
    int first                             = kind == Kind::correlator ? 1 : 0;
    full_directions(0)                    = -1;
    full_directions(nda::range(first, m)) = indices.directions;
    auto kernels                          = prepare_tau_kernel_placement(topology, full_directions, indices.pole_indices, poles);
    for (int a = 1; a < m; ++a) {
      int orbital = indices.flavours(a - 1);
      if (orbital < 0 || orbital >= flavours) throw std::invalid_argument("orbital index out of range");
      vertices[topology.arcs(a, 0)].flavour = orbital;
      vertices[topology.arcs(a, 1)].flavour = orbital;
    }
    int direction_sign = 1;
    for (int a = 1; a < m; ++a) {
      vertices[topology.arcs(a, 1)].pole_slot = a - 1;
      if (full_directions(a) == 0) direction_sign *= -1;
    }
    return {topology, std::move(vertices), std::move(full_directions), indices.pole_indices, std::move(kernels), direction_sign};
  }

  Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles, long flat_index, Kind kind) {
    auto indices = decode_indices(topology.order(), flavours, poles.size(), flat_index, kind);
    return prepare_diagram(topology, flavours, indices, poles, kind);
  }

  std::ostream &operator<<(std::ostream &out, Diagram const &diagram) {
    out << "Topology:\n"
        << diagram.topology.arcs << "\nDirections: " << diagram.directions << "\nPoles: " << diagram.pole_indices
        << "\nVertices (flavour, adjoint, bar, pole slot):\n";
    for (int v = 0; v < diagram.vertices.size(); ++v) {
      auto const &vertex = diagram.vertices[v];
      out << v << ": " << vertex.flavour << ' ' << vertex.adjoint << ' ' << vertex.barred << ' ' << vertex.pole_slot << '\n';
    }
    return out << "Edge kernels:\n"
               << diagram.kernel_placement.edges << "\nPrefactor: " << diagram.direction_sign << ", kernel signs "
               << diagram.kernel_placement.prefactor_argument_signs << ", exponents " << diagram.kernel_placement.prefactor_exponents << '\n';
  }

} // namespace triqs_xca::diagram

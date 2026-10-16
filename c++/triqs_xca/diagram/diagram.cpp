#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/diagram/arc.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

namespace triqs_xca::diagram {
  std::vector<Vertex> prepare_vertices(Topology const &topology, Arcs const &arcs, Kind kind) {
    int m = topology.order();
    arcs.validate(m, kind);
    int first = kind == Kind::correlator ? 1 : 0;
    std::vector<Vertex> result(2 * m, {false, false, -1, 0});
    // Legacy external correlator endpoint convention; statistics are assigned separately.
    result[0].adjoint = true;
    for (int a = first; a < m; ++a) {
      ArcOrientation orientation{arcs.directions(a) == 1};
      auto &early   = result[topology.arcs(a, 0)];
      auto &late    = result[topology.arcs(a, 1)];
      early.adjoint = orientation.early_operator(-1).adjoint;
      late.adjoint  = orientation.late_operator(-1).adjoint;
      late.barred   = a != 0;
      if (a != 0) {
        early.flavour = late.flavour = arcs.flavours(a - 1);
        late.pole_slot               = a - 1;
      }
    }
    return result;
  }

  Diagram prepare_diagram(Topology const &topology, int flavours, Arcs const &arcs, nda::vector_const_view<double> poles, Kind kind) {
    if (flavours < 0) throw std::invalid_argument("negative flavour count");
    auto vertices = prepare_vertices(topology, arcs, kind);
    for (int flavour : arcs.flavours)
      if (flavour >= flavours) throw std::invalid_argument("flavour index out of range");
    auto [kernels, prefactor] = prepare_tau_kernels(topology, arcs, poles, kind);
    int direction_sign        = 1;
    for (int a = 1; a < topology.order(); ++a)
      if (arcs.directions(a) == 0) direction_sign *= -1;
    return {topology, arcs, {std::move(vertices), std::move(kernels)}, std::move(prefactor), direction_sign};
  }

  Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles, long flat_index, Kind kind) {
    auto arcs = decode_arcs(topology.order(), flavours, poles.size(), flat_index, kind);
    return prepare_diagram(topology, flavours, arcs, poles, kind);
  }

  std::ostream &operator<<(std::ostream &out, Diagram const &diagram) {
    out << "Topology:\n"
        << diagram.topology.arcs << "\nDirections: " << diagram.arcs.directions << "\nPoles: " << diagram.arcs.pole_indices
        << "\nFlavours: " << diagram.arcs.flavours << "\nVertices (flavour, adjoint, bar, pole slot):\n";
    for (int v = 0; v < diagram.backbone.vertices.size(); ++v) {
      auto const &vertex = diagram.backbone.vertices[v];
      out << v << ": " << vertex.flavour << ' ' << vertex.adjoint << ' ' << vertex.barred << ' ' << vertex.pole_slot << '\n';
    }
    return out << "Edge kernels:\n"
               << diagram.backbone.kernel_placement.edges << "\nPrefactor: " << diagram.direction_sign << ", kernel signs "
               << diagram.kernel_prefactor.argument_signs << ", exponents " << diagram.kernel_prefactor.exponents << '\n';
  }

} // namespace triqs_xca::diagram

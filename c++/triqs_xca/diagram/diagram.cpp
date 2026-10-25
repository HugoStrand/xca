#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/diagram/arc.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>

namespace triqs_xca::diagram {
  Diagram prepare_diagram(Topology const &topology, int flavours, Arcs const &arcs, nda::vector_const_view<double> poles, Kind kind) {
    if (flavours < 0) throw std::invalid_argument("negative flavour count");
    auto backbone = prepare_backbone(topology, arcs, poles, kind);
    for (int a = 1; a < arcs.size(); ++a)
      if (arcs[a].retained_flavour >= flavours) throw std::invalid_argument("flavour index out of range");
    int direction_sign = 1;
    for (int a = 1; a < topology.order(); ++a)
      if (arcs[a].direction == Direction::backward) direction_sign *= -1;
    return {topology, arcs, std::move(backbone), direction_sign};
  }

  Diagram prepare_diagram(Topology const &topology, int flavours, nda::vector_const_view<double> poles, long flat_index, Kind kind) {
    auto arcs = decode_arcs(topology.order(), flavours, poles.size(), flat_index, kind);
    return prepare_diagram(topology, flavours, arcs, poles, kind);
  }

  std::ostream &operator<<(std::ostream &out, Diagram const &diagram) {
    out << "Topology:\n" << diagram.topology.arcs << "\nArcs (direction, retained flavour, pole):\n";
    for (auto const &arc : diagram.arcs) out << static_cast<int>(arc.direction) << ' ' << arc.retained_flavour << ' ' << arc.pole_index << '\n';
    out << "Vertices (arc, adjoint, bar, kernel sign):\n";
    for (auto const &vertex : diagram.backbone.vertices)
      out << vertex.arc_idx << ' ' << vertex.adjoint << ' ' << vertex.barred << ' ' << vertex.kernel_sign << '\n';
    out << "Edge kernels (arc, sign):\n";
    for (int e = 0; e < diagram.backbone.edges.size(); ++e) {
      out << e << ":";
      for (auto const &factor : diagram.backbone.edges[e].kernels) out << " (" << factor.arc_idx << ", " << factor.argument_sign << ")";
      out << '\n';
    }
    return out << "Prefactor: " << diagram.direction_sign << ", kernel signs " << diagram.backbone.kernel_prefactor.argument_signs << ", exponents "
               << diagram.backbone.kernel_prefactor.exponents << '\n';
  }

} // namespace triqs_xca::diagram

#include "diagram.hpp"
#include <stdexcept>

namespace triqs_xca::diagram {
  std::vector<Vertex> prepare_vertices(Topology const &topology, Arcs const &arcs, Kind kind) {
    int m = topology.order();
    validate_arcs(arcs, m, kind);
    std::vector<Vertex> result(2 * m);
    for (int a = 0; a < m; ++a) {
      auto &early   = result[topology.arcs(a, 0)];
      auto &late    = result[topology.arcs(a, 1)];
      early.arc_idx = late.arc_idx = a;
      if (a == 0 && kind == Kind::correlator) {
        early.adjoint = true; // external operators are supplied separately
        continue;
      }
      bool forward = arcs[a].direction == Direction::forward;
      early.adjoint = !forward;
      late.adjoint  = forward;
      late.barred   = a != 0;
    }
    return result;
  }

  Backbone prepare_backbone(Topology const &topology, Arcs const &arcs, nda::vector_const_view<double> poles, Kind kind) {
    int m          = topology.order();
    auto vertices  = prepare_vertices(topology, arcs, kind);
    int edge_count = 2 * m - (kind == Kind::self_energy ? 1 : 0);
    Backbone result{std::move(vertices), std::vector<Edge>(edge_count), {nda::zeros<int>(m), nda::zeros<int>(m)}};
    auto &prefactor = result.kernel_prefactor;
    for (int a = 1; a < m; ++a) {
      int pole = arcs[a].pole_index;
      if (pole < 0 || pole >= poles.size()) throw std::invalid_argument("pole index out of range");
      bool forward = arcs[a].direction == Direction::forward;
      int right = topology.arcs(a, 0), left = topology.arcs(a, 1);
      // Avoid exponentially growing kernels: select vertex or edge placement by
      // the direction and pole sign. Zero poles belong to the vertex branch.
      if (forward ? poles(pole) <= 0 : poles(pole) >= 0) {
        int sign                           = forward ? -1 : 1;
        result.vertices[right].kernel_sign = sign;
        result.vertices[left].kernel_sign  = -sign;
        prefactor.argument_signs(a)        = sign;
        prefactor.exponents(a)             = 1;
      } else {
        int sign = forward ? 1 : -1;
        for (int e = right; e < left; ++e) result.edges[e].kernels.push_back({a, sign});
        prefactor.argument_signs(a) = sign;
        prefactor.exponents(a)      = left - right - 1;
      }
    }
    return result;
  }

} // namespace triqs_xca::diagram

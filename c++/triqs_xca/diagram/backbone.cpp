#include "diagram.hpp"
#include "arc.hpp"
#include <stdexcept>

namespace triqs_xca::diagram {
  std::vector<Vertex> prepare_vertices(Topology const &topology, Arcs const &arcs, Kind kind) {
    int m = topology.order();
    arcs.validate(m, kind);
    int first = kind == Kind::correlator ? 1 : 0;
    std::vector<Vertex> result(2 * m, {false, false, -1, 0, {}});
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

  Backbone prepare_backbone(Topology const &topology, Arcs const &arcs, nda::vector_const_view<double> poles, Kind kind) {
    int m          = topology.order();
    auto vertices  = prepare_vertices(topology, arcs, kind);
    int edge_count = 2 * m - (kind == Kind::self_energy ? 1 : 0);
    Backbone result{std::move(vertices), std::vector<Edge>(edge_count), {nda::vector<int>(m - 1), nda::vector<int>(m - 1)}};
    auto &prefactor = result.kernel_prefactor;
    for (int a = 1; a < m; ++a) {
      int p = a - 1, pole = arcs.pole_indices(p);
      if (pole < 0 || pole >= poles.size()) throw std::invalid_argument("pole index out of range");
      int direction = arcs.directions(a);
      bool forward  = direction == 1;
      int right = topology.arcs(a, 0), left = topology.arcs(a, 1);
      result.vertices[left].kernel.pole_slot = p;
      // Avoid exponentially growing kernels: select vertex or edge placement by
      // the direction and pole sign. Zero poles belong to the vertex branch.
      if (forward ? poles(pole) <= 0 : poles(pole) >= 0) {
        int sign                                   = forward ? -1 : 1;
        result.vertices[right].kernel              = {p, sign};
        result.vertices[left].kernel.argument_sign = -sign;
        prefactor.argument_signs(p)                = sign;
        prefactor.exponents(p)                     = 1;
      } else {
        int sign = forward ? 1 : -1;
        for (int e = right; e < left; ++e) result.edges[e].kernels.push_back({a, sign});
        prefactor.argument_signs(p) = sign;
        prefactor.exponents(p)      = left - right - 1;
      }
    }
    return result;
  }

} // namespace triqs_xca::diagram

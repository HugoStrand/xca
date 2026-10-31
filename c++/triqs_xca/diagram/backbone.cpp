#include "backbone.hpp"
#include <ostream>
#include <stdexcept>

namespace triqs_xca::diagram {
  std::vector<Vertex> prepare_vertices(Topology const &topology, Arcs const &arcs) {
    int m = topology.order();
    validate_arcs(arcs, m);
    std::vector<Vertex> result(2 * m);
    for (int a = 0; a < m; ++a) {
      auto &early   = result[topology.arcs(a, 0)];
      auto &late    = result[topology.arcs(a, 1)];
      early.flavour = late.flavour = arcs[a].retained_flavour;
      early.pole_index = late.pole_index = arcs[a].pole_index;
      if (a == 0 && arcs[a].direction == Direction::external) {
        early.adjoint = true; // external operators are supplied separately
        continue;
      }
      bool forward  = arcs[a].direction == Direction::forward;
      early.adjoint = !forward;
      late.adjoint  = forward;
      late.barred   = a != 0;
    }
    return result;
  }

  Backbone prepare_backbone(Topology const &topology, Arcs const &arcs) {
    int m          = topology.order();
    auto vertices  = prepare_vertices(topology, arcs);
    int edge_count = 2 * m - (arcs[0].direction != Direction::external);
    Backbone result{std::move(vertices),
                    std::vector<Edge>(edge_count),
                    {},
                    {topology.special_vertex(), arcs[0].direction, prepare_permutation_signs(topology, arcs)}};
    result.kernel_prefactor.factors.reserve(m - 1);
    for (int a = 1; a < m; ++a) {
      auto const &arc = arcs[a];
      bool forward    = arc.direction == Direction::forward;
      int right = topology.arcs(a, 0), left = topology.arcs(a, 1);
      // Avoid exponentially growing kernels. Zero poles use vertex placement
      // in both directions; preparation needs only the selected pole's sign.
      if (forward ? arc.pole_sign <= 0 : arc.pole_sign >= 0) {
        int sign                           = forward ? -1 : 1;
        result.vertices[right].kernel_sign = sign;
        result.vertices[left].kernel_sign  = -sign;
        result.kernel_prefactor.factors.push_back({arc.pole_index, sign, 1});
      } else {
        int sign = forward ? 1 : -1;
        for (int e = right; e < left; ++e) result.edges[e].kernels.push_back({arc.pole_index, sign});
        result.kernel_prefactor.factors.push_back({arc.pole_index, sign, left - right - 1});
      }
      if (!forward) result.direction_sign *= -1;
    }
    return result;
  }

  EndpointRanges Backbone::endpoint_ranges() const {
    int special = endpoints.special_vertex;
    return {{1, special, 0, special}, {special + 1, static_cast<int>(vertices.size()), special, static_cast<int>(edges.size())}};
  }

  CorrelatorStatistics uniform_correlator_statistics(bool fermionic) {
    auto statistics = fermionic ? Statistics::fermionic : Statistics::bosonic;
    return {statistics, statistics};
  }

  void Backbone::require_self_energy() const {
    if (endpoints.direction != Direction::forward && endpoints.direction != Direction::backward)
      throw std::invalid_argument("backbone has no self-energy endpoint sum");
  }

  void Backbone::require_correlator() const {
    if (endpoints.direction != Direction::external) throw std::invalid_argument("backbone has no external observable endpoints");
  }

  int Backbone::self_energy_sign() const {
    require_self_energy();
    return (order() % 2 == 0 ? -1 : 1) * (endpoints.direction == Direction::forward ? 1 : -1) * direction_sign;
  }

  std::ostream &operator<<(std::ostream &out, Backbone const &backbone) {
    out << "Vertices (flavour, pole, adjoint, bar, kernel sign):\n";
    for (auto const &vertex : backbone.vertices)
      out << vertex.flavour << ' ' << vertex.pole_index << ' ' << vertex.adjoint << ' ' << vertex.barred << ' ' << vertex.kernel_sign << '\n';
    out << "Edge kernels (pole, sign):\n";
    for (int e = 0; e < backbone.edges.size(); ++e) {
      out << e << ":";
      for (auto const &factor : backbone.edges[e].kernels) out << " (" << factor.pole_index << ", " << factor.argument_sign << ")";
      out << '\n';
    }
    out << "Endpoint: " << backbone.endpoints.special_vertex << ", direction " << static_cast<int>(backbone.endpoints.direction) << ", parity "
        << backbone.endpoints.signs.with_fermionic_arc << ", " << backbone.endpoints.signs.without_fermionic_arc
        << "\nDirection sign: " << backbone.direction_sign << "\nKernel normalization (pole, sign, exponent):";
    for (auto const &factor : backbone.kernel_prefactor.factors)
      out << " (" << factor.pole_index << ", " << factor.argument_sign << ", " << factor.exponent << ")";
    return out << '\n';
  }
} // namespace triqs_xca::diagram

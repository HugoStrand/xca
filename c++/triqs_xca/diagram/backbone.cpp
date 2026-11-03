#include "backbone.hpp"
#include <ostream>
#include <stdexcept>

namespace triqs_xca::diagram {
  std::vector<Vertex> prepare_vertices(nda::array_const_view<int, 2> topology, Arcs const &arcs) {
    int m = topology.extent(0);
    validate_arcs(arcs, m);
    std::vector<Vertex> result(2 * m);
    for (int a = 0; a < m; ++a) {
      auto &early   = result[topology(a, 0)];
      auto &late    = result[topology(a, 1)];
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

  template <Kind K> Backbone<K> prepare_backbone(nda::array_const_view<int, 2> topology, Arcs const &arcs) {
    int m          = topology.extent(0);
    auto vertices  = prepare_vertices(topology, arcs);
    if ((arcs[0].direction == Direction::external) != (K == Kind::correlator))
      throw std::invalid_argument("special arc does not match backbone kind");
    int edge_count = 2 * m - (K == Kind::self_energy);
    Backbone<K> result(arcs[0].direction);
    result.vertices  = std::move(vertices);
    result.edges     = std::vector<Edge>(edge_count);
    result.endpoints = {topology(0, 1), prepare_permutation_signs(topology, arcs)};
    result.kernel_prefactor.factors.reserve(m - 1);
    for (int a = 1; a < m; ++a) {
      auto const &arc = arcs[a];
      bool forward    = arc.direction == Direction::forward;
      int right = topology(a, 0), left = topology(a, 1);
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

  template <Kind K> EndpointRanges Backbone<K>::endpoint_ranges() const {
    int special = endpoints.special_vertex;
    return {{1, special, 0, special}, {special + 1, static_cast<int>(vertices.size()), special, static_cast<int>(edges.size())}};
  }

  template <Kind K> OperatorPairOrientation Backbone<K>::endpoint_orientation() const requires(K == Kind::self_energy) {
    return {endpoint_direction_ == Direction::forward};
  }

  CorrelatorStatistics uniform_correlator_statistics(bool fermionic) {
    auto statistics = fermionic ? Statistics::fermionic : Statistics::bosonic;
    return {statistics, statistics};
  }

  template <Kind K> int Backbone<K>::self_energy_sign() const requires(K == Kind::self_energy) {
    return (order() % 2 == 0 ? -1 : 1) * (endpoint_direction_ == Direction::forward ? 1 : -1) * direction_sign;
  }

  template <Kind K> std::ostream &operator<<(std::ostream &out, Backbone<K> const &backbone) {
    out << "Vertices (flavour, pole, adjoint, bar, kernel sign):\n";
    for (auto const &vertex : backbone.vertices)
      out << vertex.flavour << ' ' << vertex.pole_index << ' ' << vertex.adjoint << ' ' << vertex.barred << ' ' << vertex.kernel_sign << '\n';
    out << "Edge kernels (pole, sign):\n";
    for (int e = 0; e < backbone.edges.size(); ++e) {
      out << e << ":";
      for (auto const &factor : backbone.edges[e].kernels) out << " (" << factor.pole_index << ", " << factor.argument_sign << ")";
      out << '\n';
    }
    out << "Endpoint: " << backbone.endpoints.special_vertex << ", direction " << static_cast<int>(backbone.endpoint_direction_) << ", parity "
        << backbone.endpoints.signs.with_fermionic_arc << ", " << backbone.endpoints.signs.without_fermionic_arc
        << "\nDirection sign: " << backbone.direction_sign << "\nKernel normalization (pole, sign, exponent):";
    for (auto const &factor : backbone.kernel_prefactor.factors)
      out << " (" << factor.pole_index << ", " << factor.argument_sign << ", " << factor.exponent << ")";
    return out << '\n';
  }

  template struct Backbone<Kind::self_energy>;
  template struct Backbone<Kind::correlator>;
  template SelfEnergyBackbone prepare_backbone<Kind::self_energy>(nda::array_const_view<int, 2>, Arcs const &);
  template CorrelatorBackbone prepare_backbone<Kind::correlator>(nda::array_const_view<int, 2>, Arcs const &);
  template std::ostream &operator<<(std::ostream &, SelfEnergyBackbone const &);
  template std::ostream &operator<<(std::ostream &, CorrelatorBackbone const &);
} // namespace triqs_xca::diagram

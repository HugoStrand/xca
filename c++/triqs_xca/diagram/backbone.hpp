#pragma once

#include "topology.hpp"
#include "arcs.hpp"
#include <iosfwd>

namespace triqs_xca::diagram {

  /// A resolved operator and scalar-kernel instruction. Distinguished endpoints
  /// have no fixed flavour or pole; their operation is described by Endpoints.
  struct Vertex {
    int flavour     = -1;
    int pole_index  = -1;
    bool barred     = false;
    bool adjoint    = false;
    int kernel_sign = 0;
  };

  struct KernelFactor {
    int pole_index;
    int argument_sign;
    bool operator==(KernelFactor const &) const = default;
  };

  struct Edge {
    std::vector<KernelFactor> kernels; // original arc order; an empty list is identity
  };

  struct KernelPower {
    int pole_index;
    int argument_sign;
    int exponent;
    bool operator==(KernelPower const &) const = default;
  };

  /// Whole-backbone normalization in original internal-arc order. Repeated poles
  /// remain separate terms; numerical values belong to Action.
  struct TauKernelPrefactor {
    std::vector<KernelPower> factors;
  };

  /// Half-open intervals of an alternating backbone. Vertex i connects sector
  /// slots i and i+1; edge e lies at slot e+1. Either outer vertex may be omitted.
  /// A single-edge range has no internal vertices. An empty range has neither
  /// vertices nor edges; a vertex-only range is invalid. No sector layout belongs here.
  struct BackboneRange {
    int first_vertex, vertex_end, first_edge, edge_end;

    BackboneRange(int first_vertex, int vertex_end, int first_edge, int edge_end)
       : first_vertex(first_vertex), vertex_end(vertex_end), first_edge(first_edge), edge_end(edge_end) {
      validate();
    }

    void validate() const {
      if (first_vertex < 0 || vertex_end < first_vertex || first_edge < 0 || edge_end < first_edge
          || (edge_end == first_edge && vertex_end != first_vertex) || first_vertex - first_edge < 0 || first_vertex - first_edge > 1
          || vertex_end - edge_end < 0 || vertex_end - edge_end > 1)
        throw std::invalid_argument("invalid backbone range");
    }
    bool empty() const { return first_vertex == vertex_end && first_edge == edge_end; }

    bool contains(BackboneRange const &other) const {
      return first_vertex <= other.first_vertex && other.vertex_end <= vertex_end && first_edge <= other.first_edge && other.edge_end <= edge_end;
    }
    bool operator==(BackboneRange const &) const = default;
  };

  struct EndpointRanges {
    BackboneRange before_special, after_special;
  };

  /// Distinguished vertex and endpoint parity choices. The two parity cases
  /// leave endpoint flavours unresolved.
  struct Endpoints {
    int special_vertex = -1;
    PermutationSigns signs{1, 1};
  };

  /// External operator lists have homogeneous statistics at each endpoint.
  /// Supplied per evaluation to select a prepared endpoint sign; not Backbone state.
  struct CorrelatorStatistics {
    Statistics at_zero = Statistics::fermionic;
    Statistics at_tau  = Statistics::fermionic;
  };

  /// Compatibility adapter for the existing uniform-statistics correlator API.
  CorrelatorStatistics uniform_correlator_statistics(bool fermionic);

  /// Complete diagram-specific instructions, independent of construction inputs.
  /// No topology, arc catalogue, numerical arrays or sector layout are borrowed.
  /// Reprepare after changing source choices, statistics or pole signs.
  template <Kind K> struct Backbone {
    static_assert(K == Kind::self_energy || K == Kind::correlator);

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    TauKernelPrefactor kernel_prefactor;
    Endpoints endpoints;
    int direction_sign = 1;

    int order() const { return static_cast<int>(vertices.size()) / 2; }
    BackboneRange range() const { return {0, static_cast<int>(vertices.size()), 0, static_cast<int>(edges.size())}; }
    EndpointRanges endpoint_ranges() const;
    /// Interaction endpoint orientation, available only for self-energy backbones.
    OperatorPairOrientation endpoint_orientation() const requires(K == Kind::self_energy);
    /// Global order/orientation sign, excluding endpoint coefficient reflection,
    /// permutation parity and numerical kernel normalization.
    int self_energy_sign() const requires(K == Kind::self_energy);

    private:
    Direction endpoint_direction_;
    explicit Backbone(Direction direction) : endpoint_direction_(direction) {}

    template <Kind Other> friend Backbone<Other> prepare_backbone(nda::array_const_view<int, 2> topology, Arcs const &arcs);
    template <Kind Other> friend std::ostream &operator<<(std::ostream &out, Backbone<Other> const &backbone);
  };

  using SelfEnergyBackbone = Backbone<Kind::self_energy>;
  using CorrelatorBackbone = Backbone<Kind::correlator>;

  /// Preparation functions borrow connectivity only for the duration of the call.
  /// Validate it once with topology::validate() before preparing any backbones.
  /// The special arc must match the requested backbone kind.
  template <Kind K> Backbone<K> prepare_backbone(nda::array_const_view<int, 2> topology, Arcs const &arcs);

  /// Evaluator entry point: decode the selected arc choices and prepare their backbone.
  /// Requires connectivity validated by topology::validate().
  template <Kind K, typename Storage> Backbone<K> prepare_backbone(nda::array_const_view<int, 2> topology, Action<Storage> const &action, long flat_index) {
    return prepare_backbone<K>(topology, decode_arcs(topology.extent(0), action, flat_index, K));
  }

  /// Construction helper; returned vertices do not yet have kernel placement.
  /// Requires connectivity validated by topology::validate().
  std::vector<Vertex> prepare_vertices(nda::array_const_view<int, 2> topology, Arcs const &arcs);
  template <Kind K> std::ostream &operator<<(std::ostream &out, Backbone<K> const &backbone);
} // namespace triqs_xca::diagram

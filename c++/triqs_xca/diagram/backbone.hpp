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

  /// Forward/backward denotes an interaction endpoint sum. External denotes
  /// supplied observables. The two parity cases leave endpoint flavours unresolved.
  struct Endpoints {
    int special_vertex  = -1;
    Direction direction = Direction::external;
    PermutationSigns signs{1, 1};
  };

  /// Complete diagram-specific instructions, independent of construction inputs.
  /// No topology, arc catalogue, numerical arrays or sector layout are borrowed.
  /// Reprepare after changing source choices, statistics or pole signs.
  struct Backbone {
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    TauKernelPrefactor kernel_prefactor;
    Endpoints endpoints;
    int direction_sign = 1;

    int order() const { return static_cast<int>(vertices.size()) / 2; }
    BackboneRange range() const { return {0, static_cast<int>(vertices.size()), 0, static_cast<int>(edges.size())}; }
    EndpointRanges endpoint_ranges() const;
    void require_self_energy() const;
    void require_correlator() const;
    /// Global order/orientation sign, excluding endpoint coefficient reflection,
    /// permutation parity and numerical kernel normalization.
    int self_energy_sign() const;
  };

  Backbone prepare_backbone(Topology const &topology, Arcs const &arcs);
  /// Construction helper; returned vertices do not yet have kernel placement.
  std::vector<Vertex> prepare_vertices(Topology const &topology, Arcs const &arcs);
  std::ostream &operator<<(std::ostream &out, Backbone const &backbone);
} // namespace triqs_xca::diagram

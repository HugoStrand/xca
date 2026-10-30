#pragma once

#include "topology.hpp"
#include "arcs.hpp"

namespace triqs_xca::diagram {

  /// A vertex refers to its topology arc, including the distinguished endpoints.
  /// Flavour and pole choices belong to Arcs. Only barred operators need a pole.
  struct Vertex {
    int arc_idx     = -1;
    bool barred     = false;
    bool adjoint    = false;
    int kernel_sign = 0;
  };

  struct KernelFactor {
    int arc_idx;
    int argument_sign;
    bool operator==(KernelFactor const &) const = default;
  };

  struct Edge {
    std::vector<KernelFactor> kernels; // increasing arc order; an empty list is identity
  };

  /// Whole-backbone normalization, indexed by topology arc; entry zero is identity.
  struct TauKernelPrefactor {
    nda::vector<int> argument_signs;
    nda::vector<int> exponents;
  };

  /// Half-open intervals of an alternating backbone. Vertex i connects sector
  /// slots i and i+1; edge e lies at slot e+1. Either outer vertex may be omitted.
  /// A single-edge range has no internal vertices. An empty range has neither
  /// vertices nor edges; a vertex-only range is invalid. No sector layout belongs here.
  struct BackboneRange {
    int first_vertex, vertex_end, first_edge, edge_end;

    BackboneRange(int first_vertex, int vertex_end, int first_edge, int edge_end)
       : first_vertex(first_vertex), vertex_end(vertex_end), first_edge(first_edge), edge_end(edge_end) { validate(); }

    void validate() const {
      if (first_vertex < 0 || vertex_end < first_vertex || first_edge < 0 || edge_end < first_edge || (edge_end == first_edge && vertex_end != first_vertex)
          || first_vertex - first_edge < 0 || first_vertex - first_edge > 1
          || vertex_end - edge_end < 0 || vertex_end - edge_end > 1)
        throw std::invalid_argument("invalid backbone range");
    }
    bool empty() const { return first_vertex == vertex_end && first_edge == edge_end; }

    bool contains(BackboneRange const &other) const {
      return first_vertex <= other.first_vertex && other.vertex_end <= vertex_end
          && first_edge <= other.first_edge && other.edge_end <= edge_end;
    }
    bool operator==(BackboneRange const &) const = default;
  };

  struct EndpointRanges {
    BackboneRange before_special, after_special;
  };

  /// Ordered operator vertices and local scalar-kernel prescriptions. A missing
  /// edge factor means the identity, including the trailing correlator edge.
  /// The normalization recipe belongs to the whole backbone, not each segment.
  struct Backbone {
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    TauKernelPrefactor kernel_prefactor;

    BackboneRange range() const { return {0, static_cast<int>(vertices.size()), 0, static_cast<int>(edges.size())}; }
  };
  Backbone prepare_backbone(Topology const &topology, Arcs const &arcs, nda::vector_const_view<double> poles, Kind kind);
} // namespace triqs_xca::diagram

#pragma once
#include <stdexcept>

namespace triqs_xca::diagram {
  /// Half-open intervals of an alternating backbone. Vertex i connects sector
  /// slots i and i+1; edge e lies at slot e+1. Either outer vertex may be omitted.
  /// A single-edge range has no internal vertices. No sector layout belongs here.
  struct BackboneRange {
    int first_vertex, vertex_end, first_edge, edge_end;

    BackboneRange(int first_vertex, int vertex_end, int first_edge, int edge_end)
       : first_vertex(first_vertex), vertex_end(vertex_end), first_edge(first_edge), edge_end(edge_end) { validate(); }

    void validate() const {
      if (first_vertex < 0 || vertex_end < first_vertex || first_edge < 0 || edge_end <= first_edge
          || first_vertex - first_edge < 0 || first_vertex - first_edge > 1
          || vertex_end - edge_end < 0 || vertex_end - edge_end > 1)
        throw std::invalid_argument("invalid backbone range");
    }
    bool contains(BackboneRange const &other) const {
      return first_vertex <= other.first_vertex && other.vertex_end <= vertex_end
          && first_edge <= other.first_edge && other.edge_end <= edge_end;
    }
    bool operator==(BackboneRange const &) const = default;
  };
} // namespace triqs_xca::diagram

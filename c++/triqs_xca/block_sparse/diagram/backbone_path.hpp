#pragma once
#include <nda/nda.hpp>
#include <stdexcept>

namespace triqs_xca::block_sparse::diagram {
  /// Owned sector sequence over half-open vertex/edge intervals. Sector slot i
  /// is before vertex i; edge e carries slot e+1. A single-edge segment has zero
  /// vertices and one sector. A retained path never borrows the discovery workspace.
  struct BackbonePath {
    int first_vertex, vertex_end, first_edge, edge_end;
    nda::vector<int> sectors, sector_dimensions;

    BackbonePath(int first_vertex, int vertex_end, int first_edge, int edge_end)
       : first_vertex(first_vertex), vertex_end(vertex_end), first_edge(first_edge), edge_end(edge_end) {
      if (first_vertex < 0 || vertex_end < first_vertex || first_edge < 0 || edge_end <= first_edge || first_vertex > first_edge + 1
          || vertex_end < edge_end)
        throw std::invalid_argument("invalid backbone interval");
      sectors           = nda::vector<int>(vertex_end - first_vertex + 1);
      sector_dimensions = nda::vector<int>(vertex_end - first_vertex + 1);
    }
    int &sector(int slot) { return sectors(slot - first_vertex); }
    int sector(int slot) const { return sectors(slot - first_vertex); }
    int &dimension(int slot) { return sector_dimensions(slot - first_vertex); }
    int dimension(int slot) const { return sector_dimensions(slot - first_vertex); }
    int &edge_sector(int edge) { return sector(edge + 1); }
    int edge_sector(int edge) const { return sector(edge + 1); }
  };
} // namespace triqs_xca::block_sparse::diagram

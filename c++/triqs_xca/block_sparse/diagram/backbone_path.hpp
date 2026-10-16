#pragma once
#include <nda/nda.hpp>
#include "triqs_xca/diagram/backbone_range.hpp"

namespace triqs_xca::block_sparse::diagram {
  /// A sector realization of a shared backbone range. Owned metadata never
  /// borrows the discovery workspace; global slots translate to compact storage.
  struct BackbonePath {
    triqs_xca::diagram::BackboneRange range;
    nda::vector<int> sectors, sector_dimensions;

    explicit BackbonePath(triqs_xca::diagram::BackboneRange range)
       : range(range), sectors(range.vertex_end - range.first_vertex + 1), sector_dimensions(sectors.size()) {}

    int &sector(int slot) { return sectors(slot - range.first_vertex); }
    int sector(int slot) const { return sectors(slot - range.first_vertex); }
    int &dimension(int slot) { return sector_dimensions(slot - range.first_vertex); }
    int dimension(int slot) const { return sector_dimensions(slot - range.first_vertex); }
    int &edge_sector(int edge) { return sector(edge + 1); }
    int edge_sector(int edge) const { return sector(edge + 1); }
  };
} // namespace triqs_xca::block_sparse::diagram

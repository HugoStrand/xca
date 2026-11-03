#pragma once
#include <nda/nda.hpp>
#include "triqs_xca/diagram/backbone.hpp"

namespace triqs_xca::block_sparse::diagram {
  /// Owned instructions and sector realization for a complete contribution or
  /// selected range. Copying retains no discovery workspace or numerical views.
  template <triqs_xca::diagram::Kind K> struct BackbonePath {
    triqs_xca::diagram::Backbone<K> backbone;
    triqs_xca::diagram::BackboneRange range;
    nda::vector<int> sectors, sector_dimensions;
    /// Ordered Action flavour IDs for an interaction sum, or observable-list
    /// indices for external endpoints. Uncompleted segments leave these empty;
    /// an empty endpoint selection never stands for an identity contribution.
    std::vector<int> zero_operators, special_operators;

    BackbonePath(triqs_xca::diagram::Backbone<K> const &backbone, triqs_xca::diagram::BackboneRange range) : backbone(backbone), range(range) {
      range.validate();
      if (!backbone.range().contains(range)) throw std::invalid_argument("path range is outside backbone");
      sectors           = nda::vector<int>(range.vertex_end - range.first_vertex + 1);
      sector_dimensions = nda::vector<int>(sectors.size());
      sectors           = -1;
      sector_dimensions = -1;
    }

    explicit BackbonePath(triqs_xca::diagram::Backbone<K> const &backbone) : BackbonePath(backbone, backbone.range()) {}

    int &sector(int slot) { return sectors(slot - range.first_vertex); }
    int sector(int slot) const { return sectors(slot - range.first_vertex); }
    int &dimension(int slot) { return sector_dimensions(slot - range.first_vertex); }
    int dimension(int slot) const { return sector_dimensions(slot - range.first_vertex); }
    int &edge_sector(int edge) { return sector(edge + 1); }
    int edge_sector(int edge) const { return sector(edge + 1); }
  };

  using SelfEnergyBackbonePath = BackbonePath<triqs_xca::diagram::Kind::self_energy>;
  using CorrelatorBackbonePath = BackbonePath<triqs_xca::diagram::Kind::correlator>;
} // namespace triqs_xca::block_sparse::diagram

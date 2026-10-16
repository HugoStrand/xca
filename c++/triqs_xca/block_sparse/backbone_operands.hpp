#pragma once
#include <optional>
#include <vector>
#include "diagram/backbone_path.hpp"

namespace triqs_xca::block_sparse {
  /// CPU numerical views over one backbone interval. Metadata is owned; numerical
  /// owners must remain alive and unchanged during execution. Empty vertices are
  /// supplied by completion operands, never interpreted as zero operators.
  struct BackboneOperands {
    diagram::BackbonePath path;
    std::vector<std::optional<nda::array_const_view<nda::dcomplex, 2>>> vertices;
    std::vector<nda::array_const_view<nda::dcomplex, 3>> propagators;

    nda::array_const_view<nda::dcomplex, 2> vertex(int v) const { return vertices[v - path.range.first_vertex].value(); }
    nda::array_const_view<nda::dcomplex, 3> propagator(int e) const { return propagators[e - path.range.first_edge]; }
  };
} // namespace triqs_xca::block_sparse

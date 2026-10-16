#pragma once
#include <nda/nda.hpp>

namespace triqs_xca::diagram {

  enum class Kind { self_energy, correlator };

  /// Choices attached to the connections in Topology; no matrix data or execution order.
  /// directions(a) uses topology row a, with 0/1 for backward/forward and -1
  /// only at row 0 of a correlator (its external arc is absent).
  /// pole_indices(a-1) and flavours(a-1) describe internal rows a >= 1.
  /// External flavours are supplied by the endpoint sum or correlator operators.
  struct Arcs {
    nda::vector<int> directions;
    nda::vector<int> pole_indices;
    nda::vector<int> flavours;

    /// Check layout and nonnegative internal indices; model bounds are checked
    /// during preparation against the flavour count and numerical pole table.
    void validate(int order, Kind kind) const;
  };

  /// The legacy flat-index convention lives at the selection boundary.
  long count_diagrams(int order, int flavours, long pole_count, Kind kind);
  Arcs decode_arcs(int order, int flavours, long pole_count, long flat_index, Kind kind);
} // namespace triqs_xca::diagram

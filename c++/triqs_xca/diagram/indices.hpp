#pragma once
#include <nda/nda.hpp>

namespace triqs_xca::diagram {

enum class Kind { self_energy, correlator };

/// Logical indices: one direction per interaction arc (the external correlator
/// arc is absent), one pole and one orbital per internal arc. No linear index
/// or execution order is needed to prepare these assignments.
struct Indices {
  nda::vector<int> directions;
  nda::vector<int> pole_indices;
  nda::vector<int> flavours;
};

/// The legacy flat-index convention lives at the selection boundary.
long count_diagrams(int order, int flavours, long pole_count, Kind kind);
Indices decode_indices(int order, int flavours, long pole_count, long flat_index, Kind kind);
} // namespace triqs_xca::diagram

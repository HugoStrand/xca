#pragma once
#include "backbone_path.hpp"

namespace triqs_xca::block_sparse::diagram {
  /// One block realization of a shared Diagram. Endpoint sets and output policy
  /// belong here, not in the generic sector path.
  struct SelfEnergyDiagram {
    BackbonePath path;
    int early_symmetry_set = 0, late_symmetry_set = 0;
    int output_sector = 0;
    explicit SelfEnergyDiagram(int order) : path({0, 2 * order, 0, 2 * order - 1}) {}
  };

} // namespace triqs_xca::block_sparse::diagram

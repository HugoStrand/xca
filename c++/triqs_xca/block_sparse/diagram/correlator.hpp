#pragma once
#include <vector>

namespace triqs_xca::block_sparse::diagram {
  /// Completion choices for two retained backbone paths. Indices refer to the
  /// caller's path/result tables for one shared Diagram; no borrowed matrix views.
  struct CorrelatorDiagram {
    int zero_to_tau_path, tau_to_beta_path;
    int at_tau_operator;
    std::vector<int> at_zero_operators;
  };

} // namespace triqs_xca::block_sparse::diagram

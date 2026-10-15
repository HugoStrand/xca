#pragma once
#include <vector>
#include "topology.hpp"

namespace triqs_xca::diagram {
  struct VertexTauKernel {
    int pole_slot     = 0;
    int argument_sign = 0;
  };
  /// Placement only: no operator matrices, numerical kernel tables or discrete signs.
  struct TauKernelPlacement {
    std::vector<VertexTauKernel> vertices;
    nda::array<int, 2> edges;
    nda::vector<int> prefactor_argument_signs;
    nda::vector<int> prefactor_exponents;
  };
  TauKernelPlacement prepare_tau_kernel_placement(Topology const &topology, nda::vector_const_view<int> directions,
                                                  nda::vector_const_view<int> pole_indices, nda::vector_const_view<double> poles);
} // namespace triqs_xca::diagram

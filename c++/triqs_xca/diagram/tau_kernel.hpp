#pragma once
#include <vector>
#include "topology.hpp"
#include "arcs.hpp"
#include <utility>

namespace triqs_xca::diagram {
  struct VertexTauKernel {
    int pole_slot     = 0;
    int argument_sign = 0;
  };
  /// Placement only: no operator matrices, numerical kernel tables or discrete signs.
  struct TauKernelPlacement {
    std::vector<VertexTauKernel> vertices;
    nda::array<int, 2> edges;
  };
  /// Recipe for the scalar factor of the whole diagram, separate from local kernels.
  struct TauKernelPrefactor {
    nda::vector<int> argument_signs;
    nda::vector<int> exponents;
  };
  std::pair<TauKernelPlacement, TauKernelPrefactor> prepare_tau_kernels(Topology const &topology, Arcs const &arcs,
                                                                        nda::vector_const_view<double> poles, Kind kind);
} // namespace triqs_xca::diagram

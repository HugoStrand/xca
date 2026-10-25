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
  struct KernelFactor {
    int arc_idx;
    int argument_sign;
    bool operator==(KernelFactor const &) const = default;
  };
  /// Recipe for the scalar factor of the whole diagram, separate from local kernels.
  struct TauKernelPrefactor {
    nda::vector<int> argument_signs;
    nda::vector<int> exponents;
  };
} // namespace triqs_xca::diagram

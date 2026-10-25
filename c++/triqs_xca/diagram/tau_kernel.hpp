#pragma once
#include <vector>
#include "topology.hpp"
#include "arcs.hpp"
#include <utility>

namespace triqs_xca::diagram {
  struct KernelFactor {
    int arc_idx;
    int argument_sign;
    bool operator==(KernelFactor const &) const = default;
  };
  /// Whole-backbone normalization, indexed by topology arc; entry zero is identity.
  struct TauKernelPrefactor {
    nda::vector<int> argument_signs;
    nda::vector<int> exponents;
  };
} // namespace triqs_xca::diagram

#pragma once
#include <nda/nda.hpp>

namespace triqs_xca {
  /// Precomputed numerical time-kernel tables. Plus/minus selects the beta-scaled pole argument.
  struct TauKernel {
    nda::vector<double> at_zero_plus, at_zero_minus;
    nda::array<double, 2> at_tau_plus, at_tau_minus;
  };
} // namespace triqs_xca

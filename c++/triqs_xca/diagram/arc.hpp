#pragma once

#include "triqs_xca/action.hpp"

namespace triqs_xca::diagram {

  /// Transitional endpoint description, pending integration into Backbone.
  /// Both endpoint flavours are summed over [0, flavours), with no selected pole.
  struct SelfEnergyEndpointSum {
    OperatorPairOrientation orientation;
    int flavours;
  };

} // namespace triqs_xca::diagram

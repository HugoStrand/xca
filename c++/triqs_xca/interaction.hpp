#pragma once

#include <utility>

namespace triqs_xca {

/// A model operator, before choosing dense or block-sparse storage. Adjoint acts
/// on the operator, never on the scalar interaction coefficient.
struct OperatorReference {
  int flavour;
  bool adjoint;
};

/// One directed interaction between an earlier and a later backbone vertex.
/// Forward: O_late^dag C(late, early; tau) O_early.
/// Backward: -O_late C(early, late; beta-tau) O_early^dag.
/// The minus is the existing reflected-kernel convention for both interaction
/// families. Permutation statistics and diagram-order signs are separate.
struct InteractionLine {
  bool forward;

  OperatorReference early_operator(int flavour) const { return {flavour, !forward}; }
  OperatorReference late_operator(int flavour) const { return {flavour, forward}; }
  std::pair<int, int> coefficient_indices(int late, int early) const {
    return forward ? std::pair{late, early} : std::pair{early, late};
  }
  bool reflected() const { return !forward; }
  int reflection_sign() const { return forward ? 1 : -1; }
};

struct WeightedOperator {
  OperatorReference op;
  int pole;
  int coefficient_row;
  int coefficient_column;
  int sign;
};

/// A partial contraction at the later vertex. The earlier flavour and pole are
/// fixed; the later flavour is still summed over the interaction space. A
/// representation may materialize this sum or bind its terms for execution.
/// It must not interpret the retained flavour as the later operator's flavour.
struct BarredOperator {
  InteractionLine line;
  int retained_flavour;
  int pole;

  WeightedOperator term(int summed_flavour) const {
    auto [row, column] = line.coefficient_indices(summed_flavour, retained_flavour);
    return {line.late_operator(summed_flavour), pole, row, column, line.reflection_sign()};
  }
};

/// The special self-energy arc leaves both endpoint flavours summed over
/// [0, flavours). This describes the sum without expanding its Cartesian
/// product. A block planner may restrict candidates using exact support.
/// Unlike BarredOperator, its coefficient is a time-dependent function and
/// carries no assigned pole.
struct EndpointSum {
  InteractionLine line;
  int flavours;
};

} // namespace triqs_xca

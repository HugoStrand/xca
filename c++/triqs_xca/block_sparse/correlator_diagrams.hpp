#pragma once
#include <functional>
#include <span>
#include "diagram/correlator.hpp"
#include "diagram/backbone_path.hpp"
#include "block_op.hpp"

namespace triqs_xca::block_sparse {
  /// Connectivity-only completion, in external tau/start/zero-operator order.
  /// Both path tables must come from the matching Diagram and unchanged inputs.
  void visit_correlator_diagrams(BlockDiagOpFun const &propagator, std::span<diagram::BackbonePath const> zero_to_tau,
                                 std::span<diagram::BackbonePath const> tau_to_beta, std::span<BlockOp const> at_tau, std::span<BlockOp const> at_zero,
                                 std::function<void(diagram::CorrelatorDiagram const &)> const &visit);
} // namespace triqs_xca::block_sparse

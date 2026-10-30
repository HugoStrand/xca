#pragma once
#include <functional>
#include <span>
#include "diagram/backbone_path.hpp"
#include "block_op.hpp"

namespace triqs_xca::block_sparse {
  /// Connectivity-only completion in tau/start/zero-operator order. Both tables
  /// must describe the matching endpoint ranges of the same Backbone and inputs.
  /// Callback paths own their complete instructions and endpoint selections;
  /// the two result-table indices are optional CPU scheduling information only.
  /// Copy a callback path to retain it independently of both input tables.
  void visit_correlator_paths(BlockDiagOpFun const &propagator, std::span<diagram::BackbonePath const> zero_to_tau,
                              std::span<diagram::BackbonePath const> tau_to_beta, std::span<BlockOp const> at_tau, std::span<BlockOp const> at_zero,
                              std::function<void(diagram::BackbonePath const &, int before_result, int after_result)> const &visit);
} // namespace triqs_xca::block_sparse

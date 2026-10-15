#pragma once
#include <functional>
#include <span>
#include "backbone_path.hpp"
#include "triqs_xca/diagram/correlator.hpp"
#include "triqs_xca/block_sparse/operator_support.hpp"

namespace triqs_xca::block_sparse::diagram {
  /// Completion choices for two retained backbone paths. Indices refer to the
  /// caller's path/result tables for one shared Diagram; no borrowed matrix views.
  struct CorrelatorDiagram {
    int zero_to_tau_path, tau_to_beta_path;
    int at_tau_operator;
    std::vector<int> at_zero_operators;
  };

  /// Stream paths in start-sector order. Copy a path to retain it independently
  /// of workspace reuse. Keep the existing propagator-mask placement unchanged.
  void visit_correlator_paths(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram, BlockOpSymQuartet const &operators,
                              OperatorSupportTable const &support, triqs_xca::diagram::CorrelatorSegment segment,
                              std::function<void(BackbonePath const &)> const &visit);

  /// Connectivity-only completion, in external tau/start/zero-operator order.
  /// Both path tables must come from the matching Diagram and unchanged inputs.
  void visit_correlator_diagrams(BlockDiagOpFun const &propagator, std::span<BackbonePath const> zero_to_tau,
                                 std::span<BackbonePath const> tau_to_beta, std::span<BlockOp const> at_tau, std::span<BlockOp const> at_zero,
                                 std::function<void(CorrelatorDiagram const &)> const &visit);
} // namespace triqs_xca::block_sparse::diagram

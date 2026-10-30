#pragma once
#include <functional>
#include "diagram/backbone_path.hpp"
#include "operator_support.hpp"

namespace triqs_xca::block_sparse {
  /// Discover owning paths with ordered, exactly supported endpoint flavours.
  /// Callback references name one reusable local workspace; copy to retain a path.
  /// Reads candidate group maps, internal/endpoint support and current G masks.
  /// Support must describe the supplied unchanged quartet and barred operators.
  /// Preserves group/within-group order and equal-group filtering; records rather
  /// than enforces terminal/output-sector equality. Empty endpoint lists yield
  /// no endpoint contribution. A callback must not mutate the discovery inputs.
  void visit_self_energy_paths(BlockDiagOpFun const &propagator, triqs_xca::diagram::Backbone const &backbone, BlockOpSymQuartet const &operators,
                               OperatorSupportTable const &support, std::function<void(diagram::BackbonePath const &)> const &visit);
} // namespace triqs_xca::block_sparse

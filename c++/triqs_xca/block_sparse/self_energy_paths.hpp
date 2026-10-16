#pragma once
#include <functional>
#include "diagram/self_energy.hpp"
#include "operator_support.hpp"

namespace triqs_xca::block_sparse {
  /// Discover paths for an immutable diagram assignment, without evaluating diagrams.
  /// The caller supplies one reusable path workspace of the matching order; the
  /// callback can copy it, but must not retain references into that workspace.
  /// Discovery reads candidate group maps, exact internal-operator support, and the current propagator zero mask.
  /// The support must describe the supplied quartet, including its current barred operators.
  /// It preserves legacy ordering and equal-set filtering, and records rather than
  /// enforces terminal/output-sector equality. A callback must not change the
  /// assignment, connectivity, or masks. Independently owned copies may be
  /// prepared or replaced while a retained assignment is being visited.
  void visit_self_energy_paths(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram, BlockOpSymQuartet const &operators,
                               OperatorSupportTable const &support, diagram::SelfEnergyDiagram &workspace,
                               std::function<void(diagram::SelfEnergyDiagram const &)> const &visit);

} // namespace triqs_xca::block_sparse

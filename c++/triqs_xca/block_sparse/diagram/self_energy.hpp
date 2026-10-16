#pragma once
#include <functional>
#include "backbone_path.hpp"
#include "triqs_xca/block_sparse/operator_support.hpp"

namespace triqs_xca::block_sparse::diagram {
  /// One block realization of a shared Diagram. Endpoint sets and output policy
  /// belong here, not in the generic sector path.
  struct SelfEnergyDiagram {
    BackbonePath path;
    int early_symmetry_set = 0, late_symmetry_set = 0;
    int output_sector = 0;
    explicit SelfEnergyDiagram(int order) : path({0, 2 * order, 0, 2 * order - 1}) {}
  };

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
                               OperatorSupportTable const &support, SelfEnergyDiagram &workspace,
                               std::function<void(SelfEnergyDiagram const &)> const &visit);

} // namespace triqs_xca::block_sparse::diagram

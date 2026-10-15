#pragma once

#include <functional>
#include "operator_support.hpp"

#include "triqs_xca/diagram/diagram.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"

namespace triqs_xca::block_sparse {

/// Sector metadata for one self-energy contribution. Arrays are owned; copying
/// a path retains it independently of the walker's next callback.
/// This is not yet a complete numerical task: operator/pole assignments still
/// belong to the caller's diagram assignment and quartet.
struct SelfEnergyPath {
  long flat_index = 0;
  int output_sector = 0;
  int terminal_sector = 0;
  int kap_set = 0;
  int mu_set = 0;
  int special_vertex = 0;
  nda::vector<int> propagator_blocks; // edge order, length 2*m-1
  nda::vector<int> dimensions;        // sectors before/after each vertex, length 2*m+1

  explicit SelfEnergyPath(int order) : propagator_blocks(2 * order - 1), dimensions(2 * order + 1) {}
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
void visit_self_energy_paths(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment, BlockOpSymQuartet const &operators, OperatorSupports const &support,
                             SelfEnergyPath &workspace, std::function<void(SelfEnergyPath const &)> const &visit);

} // namespace triqs_xca::block_sparse

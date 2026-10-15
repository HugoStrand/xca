#pragma once

#include <functional>

#include "triqs_xca/backbone.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"

namespace triqs_xca::block_sparse {

/// Sector metadata for one self-energy contribution. Arrays are owned; copying
/// a path retains it independently of the walker's next callback.
/// This is not yet a complete numerical task: operator/pole assignments still
/// belong to the caller's backbone and quartet.
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

/// Discover paths for an already assigned backbone, without evaluating diagrams.
/// The caller supplies one reusable path workspace of the matching order; the
/// callback can copy it, but must not retain references into that workspace.
/// Discovery reads quartet connectivity and the current propagator zero mask.
/// It preserves legacy ordering and equal-set filtering, and records rather than
/// enforces terminal/output-sector equality. The legacy getters require a mutable
/// backbone reference; discovery itself neither assigns nor resets its indices.
/// A callback must not change connectivity, masks, or backbone traversal inputs.
void visit_self_energy_paths(BlockDiagOpFun const &propagator, Backbone &backbone, BlockOpSymQuartet const &operators,
                             SelfEnergyPath &workspace, std::function<void(SelfEnergyPath const &)> const &visit);

} // namespace triqs_xca::block_sparse

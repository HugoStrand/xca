#pragma once

#include "triqs_xca/block_sparse/block_op.hpp"
#include "triqs_xca/diagram/diagram.hpp"

namespace triqs_xca::block_sparse::bind {

  /// Resolve one internal vertex in a known path; no connectivity search occurs.
  nda::array_const_view<dcomplex, 2> vertex(BlockOpSymQuartet const &operators, triqs_xca::diagram::Diagram const &diagram, int vertex,
                                            int source_sector);

} // namespace triqs_xca::block_sparse::bind

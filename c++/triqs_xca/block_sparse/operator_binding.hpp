#pragma once

#include "block_op.hpp"
#include "triqs_xca/diagram_assignment.hpp"

namespace triqs_xca::block_sparse {

/// Resolve one internal vertex in a known path; no connectivity search occurs.
nda::array_const_view<dcomplex, 2> bind_vertex_operator(BlockOpSymQuartet const &operators, DiagramAssignment const &assignment,
                                                      int vertex, int source_sector);

} // namespace triqs_xca::block_sparse

#pragma once

#include <functional>
#include "operator_support.hpp"
#include <span>
#include "operator_binding.hpp"
#include "diagram/backbone_path.hpp"
#include "triqs_xca/diagram/correlator.hpp"

namespace triqs_xca::block_sparse {

/// Borrowed operands: execute before replacing/mutating the propagator or
/// operator owner, with the matching assignment and mesh. Metadata is owned.
struct CorrelatorFragmentOperands {
  diagram::BackbonePath path;
  std::vector<nda::array_const_view<dcomplex, 2>> vertices;
  std::vector<nda::array_const_view<dcomplex, 3>> propagators;
};

/// Discover one side in start-sector order, without matrix arithmetic. Callback
/// references refer to reusable workspace; copying a path retains its metadata.
/// Current propagator-mask placement is preserved: the walk checks reached
/// sectors, with no new starting-edge check. Joins apply their existing endpoint
/// checks separately. Internal vertices use exact support prepared for this quartet.
void visit_correlator_fragments(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment,
                                BlockOpSymQuartet const &operators, OperatorSupports const &support, triqs_xca::diagram::CorrelatorSegment segment,
                                std::function<void(diagram::BackbonePath const &)> const &visit);
CorrelatorFragmentOperands bind_correlator_fragment(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment,
                                                    BlockOpSymQuartet const &operators, diagram::BackbonePath const &path);

struct CorrelatorTrace {
  int kap; // output column in observable space
  nda::array_const_view<dcomplex, 2> op;
};

/// A compatible fragment pair and external mu operand, with all closing kap
/// operators prescribed. The external-operator views borrow their input owners.
/// One join is streamed at a time; no table of every observable pair is stored.
struct CorrelatorJoin {
  int mu; // output row in observable space
  int right_start, left_start;
  nda::array_const_view<dcomplex, 2> mu_op;
  std::vector<CorrelatorTrace> traces;
};

/// End-sector maps are indexed by fragment start sector; -1 denotes absence.
/// This reads connectivity only, and retains the existing mu/start/kap order.
void visit_correlator_joins(BlockDiagOpFun const &propagator, nda::vector_const_view<int> right_ends,
                           nda::vector_const_view<int> left_ends, std::span<BlockOp const> mu_ops,
                           std::span<BlockOp const> kap_ops, std::function<void(CorrelatorJoin const &)> const &visit);

} // namespace triqs_xca::block_sparse

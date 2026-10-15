#pragma once

#include <span>
#include <vector>
#include "block_op.hpp"
#include "triqs_xca/diagram_assignment.hpp"

namespace triqs_xca::block_sparse {

struct BlockConnection { int source, target; };

/// Owned logical support. Multiple targets per source are representable here;
/// the current CPU path executor explicitly requires single-target operators.
class OperatorSupport {
  std::vector<int> offsets_, targets_;
public:
  OperatorSupport(int sectors, std::vector<BlockConnection> components);
  std::span<int const> targets(int source) const;
  int single_target(int source) const; // -1 for absent, throws for multiple targets
};

/// Exact support of the materialized operators, independent of group candidate
/// maps. Construct once per model/coefficient replacement, never per flat index.
/// It owns metadata only; rebuilding is required after changing the quartet.
struct OperatorSupports {
  std::vector<OperatorSupport> plain, adjoint, barred_adjoint, reflected_bar;
  int poles;
  explicit OperatorSupports(BlockOpSymQuartet const &operators);
  OperatorSupport const &vertex(DiagramAssignment const &assignment, int vertex) const;
};

} // namespace triqs_xca::block_sparse

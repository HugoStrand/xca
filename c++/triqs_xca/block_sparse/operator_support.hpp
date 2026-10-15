#pragma once

#include <span>
#include <vector>
#include "block_op.hpp"
#include "triqs_xca/diagram/diagram.hpp"

namespace triqs_xca::block_sparse {

struct SectorTransition { int source, target; };

/// Owned logical support. Multiple targets per source are representable here;
/// the current CPU path executor explicitly requires single-target operators.
class OperatorSupport {
  std::vector<int> offsets_, targets_;
public:
  OperatorSupport(int sectors, std::vector<SectorTransition> components);
  std::span<int const> targets(int source) const;
  int single_target(int source) const; // -1 for absent, throws for multiple targets
};

/// Exact support of the materialized operators, independent of group candidate
/// maps. Construct once per model/coefficient replacement, never per flat index.
/// It owns metadata only; rebuilding is required after changing the quartet.
struct OperatorSupportTable {
  std::vector<OperatorSupport> plain, adjoint, barred_adjoint, reflected_bar;
  int poles;
  explicit OperatorSupportTable(BlockOpSymQuartet const &operators);
  OperatorSupport const &vertex(triqs_xca::diagram::Diagram const &diagram, int vertex) const;
};

} // namespace triqs_xca::block_sparse

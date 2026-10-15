#include "correlator_plan.hpp"

namespace triqs_xca::block_sparse {

void visit_correlator_fragments(BlockDiagOpFun const &propagator, DiagramAssignment const &assignment,
                                BlockOpSymQuartet const &operators, CorrelatorSide side,
                                std::function<void(CorrelatorFragment const &)> const &visit) {
  int m = assignment.topology.order(), special = assignment.topology.special_vertex();
  int first = side == CorrelatorSide::right ? 0 : special;
  int last = side == CorrelatorSide::right ? special - 1 : 2 * m - 1;
  CorrelatorFragment path{side, first, last, -1, -1, nda::vector<int>(2 * m), nda::vector<int>(2 * m + 1)};
  path.propagator_blocks = -1;
  path.dimensions = 0;
  auto op = [&](int vertex) -> BlockOpSymSet const & {
    auto const &v = assignment.vertices[vertex];
    int set = operators.sym_set_labels(v.orbital);
    return v.dagger ? operators.F_dags[set] : operators.Fs[set];
  };
  for (int start = 0; start < propagator.get_num_block_cols(); ++start) {
    int sector = start;
    bool incomplete = false;
    path.propagator_blocks(first) = sector;
    for (int v = first + 1; v <= last; ++v) {
      sector = op(v).get_block_index(sector);
      incomplete = sector == -1 || propagator.get_zero_block_index(sector) == -1;
      if (incomplete) break;
      path.propagator_blocks(v) = sector;
    }
    if (incomplete) continue;
    path.start_sector = start;
    path.end_sector = sector;
    path.dimensions(first + 1) = propagator.get_block_size(start);
    for (int v = first + 1; v <= last; ++v)
      path.dimensions(v + 1) = op(v).get_block_size(path.propagator_blocks(v - 1), 0);
    visit(path);
  }
}

CorrelatorFragmentOperands bind_correlator_fragment(BlockDiagOpFun const &propagator, DiagramAssignment const &assignment,
                                                    BlockOpSymQuartet const &operators, CorrelatorFragment const &path) {
  CorrelatorFragmentOperands result{path, {}, {}};
  int n = 2 * assignment.topology.order();
  result.vertices.reserve(n);
  result.propagators.reserve(n);
  for (int v = 0; v < n; ++v) {
    if (v > path.first_edge && v <= path.last_edge)
      result.vertices.push_back(bind_vertex_operator(operators, assignment, v, path.propagator_blocks(v - 1)));
    else result.vertices.emplace_back();
    if (v >= path.first_edge && v <= path.last_edge) result.propagators.push_back(propagator.get_block(path.propagator_blocks(v)));
    else result.propagators.emplace_back();
  }
  return result;
}

void visit_correlator_joins(BlockDiagOpFun const &propagator, nda::vector_const_view<int> right_ends,
                           nda::vector_const_view<int> left_ends, std::span<BlockOp const> mu_ops,
                           std::span<BlockOp const> kap_ops, std::function<void(CorrelatorJoin const &)> const &visit) {
  auto incomplete = [&](int sector) { return sector == -1 || propagator.get_zero_block_index(sector) == -1; };
  for (int mu = 0; mu < mu_ops.size(); ++mu) {
    for (int start = 0; start < right_ends.size(); ++start) {
      int right_end = right_ends(start);
      if (incomplete(right_end)) continue;
      int left_start = mu_ops[mu].get_block_index(right_end);
      if (incomplete(left_start)) continue;
      int left_end = left_ends(left_start);
      if (incomplete(left_end)) continue;
      CorrelatorJoin join{mu, start, left_start, mu_ops[mu].get_block(right_end), {}};
      for (int kap = 0; kap < kap_ops.size(); ++kap)
        if (kap_ops[kap].get_block_index(left_end) == start) join.traces.push_back({kap, kap_ops[kap].get_block(left_end)});
      visit(join);
    }
  }
}

} // namespace triqs_xca::block_sparse

#include "correlator_plan.hpp"

namespace triqs_xca::block_sparse {

void visit_correlator_fragments(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment,
                                BlockOpSymQuartet const &operators, OperatorSupports const &support, triqs_xca::diagram::CorrelatorSegment segment,
                                std::function<void(diagram::BackbonePath const &)> const &visit) {
  int m = assignment.topology.order(), special = assignment.topology.special_vertex();
  int first = segment == triqs_xca::diagram::CorrelatorSegment::zero_to_tau ? 0 : special;
  int last = segment == triqs_xca::diagram::CorrelatorSegment::zero_to_tau ? special - 1 : 2 * m - 1;
  diagram::BackbonePath path(first + 1, last + 1, first, last + 1);
  auto op = [&](int vertex) -> BlockOpSymSet const & {
    auto const &v = assignment.vertices[vertex];
    int set = operators.sym_set_labels(v.flavour);
    return v.adjoint ? operators.F_dags[set] : operators.Fs[set];
  };
  for (int start = 0; start < propagator.get_num_block_cols(); ++start) {
    int sector = start;
    bool incomplete = false;
    path.edge_sector(first) = sector;
    for (int v = first + 1; v <= last; ++v) {
      sector = support.vertex(assignment, v).single_target(sector);
      incomplete = sector == -1 || propagator.get_zero_block_index(sector) == -1;
      if (incomplete) break;
      path.edge_sector(v) = sector;
    }
    if (incomplete) continue;
    path.dimension(first + 1) = propagator.get_block_size(start);
    for (int v = first + 1; v <= last; ++v)
      path.dimension(v + 1) = op(v).get_block_size(path.edge_sector(v - 1), 0);
    visit(path);
  }
}

CorrelatorFragmentOperands bind_correlator_fragment(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment,
                                                    BlockOpSymQuartet const &operators, diagram::BackbonePath const &path) {
  CorrelatorFragmentOperands result{path, {}, {}};
  int n = 2 * assignment.topology.order();
  result.vertices.reserve(n);
  result.propagators.reserve(n);
  for (int v = 0; v < n; ++v) {
    if (v >= path.first_vertex && v < path.vertex_end)
      result.vertices.push_back(bind_vertex_operator(operators, assignment, v, path.edge_sector(v - 1)));
    else result.vertices.emplace_back();
    if (v >= path.first_edge && v < path.edge_end) result.propagators.push_back(propagator.get_block(path.edge_sector(v)));
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

#include "correlator.hpp"

namespace triqs_xca::block_sparse::diagram {
void visit_correlator_paths(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment,
                                BlockOpSymQuartet const &operators, OperatorSupports const &support, triqs_xca::diagram::CorrelatorSegment segment,
                                std::function<void(BackbonePath const &)> const &visit) {
  int m = assignment.topology.order(), special = assignment.topology.special_vertex();
  int first = segment == triqs_xca::diagram::CorrelatorSegment::zero_to_tau ? 0 : special;
  int last = segment == triqs_xca::diagram::CorrelatorSegment::zero_to_tau ? special - 1 : 2 * m - 1;
  BackbonePath path(first + 1, last + 1, first, last + 1);
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

void visit_correlator_diagrams(BlockDiagOpFun const &propagator,
    std::span<BackbonePath const> zero_to_tau, std::span<BackbonePath const> tau_to_beta,
    std::span<BlockOp const> at_tau, std::span<BlockOp const> at_zero,
    std::function<void(CorrelatorDiagram const &)> const &visit) {
  std::vector<int> beta_path_by_start(propagator.get_num_block_cols(), -1);
  for (int i = 0; i < tau_to_beta.size(); ++i)
    beta_path_by_start[tau_to_beta[i].sector(tau_to_beta[i].first_vertex)] = i;
  auto incomplete = [&](int sector) { return sector == -1 || propagator.get_zero_block_index(sector) == -1; };
  for (int mu = 0; mu < at_tau.size(); ++mu) {
    for (int i = 0; i < zero_to_tau.size(); ++i) {
      auto const &right = zero_to_tau[i];
      int right_end = right.sector(right.vertex_end);
      if (incomplete(right_end)) continue;
      int left_start = at_tau[mu].get_block_index(right_end);
      if (incomplete(left_start)) continue;
      int j = beta_path_by_start[left_start];
      if (j == -1) continue;
      auto const &left = tau_to_beta[j];
      int left_end = left.sector(left.vertex_end);
      if (incomplete(left_end)) continue;
      CorrelatorDiagram result{i, j, mu, {}};
      for (int kap = 0; kap < at_zero.size(); ++kap)
        if (at_zero[kap].get_block_index(left_end) == right.sector(right.first_vertex)) result.at_zero_operators.push_back(kap);
      visit(result);
    }
  }
}
} // namespace triqs_xca::block_sparse::diagram

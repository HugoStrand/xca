#include "correlator_diagrams.hpp"

namespace triqs_xca::block_sparse {
  void visit_correlator_diagrams(BlockDiagOpFun const &propagator, std::span<diagram::BackbonePath const> zero_to_tau,
                                 std::span<diagram::BackbonePath const> tau_to_beta, std::span<BlockOp const> at_tau, std::span<BlockOp const> at_zero,
                                 std::function<void(diagram::CorrelatorDiagram const &)> const &visit) {
    std::vector<int> beta_path_by_start(propagator.get_num_block_cols(), -1);
    for (int i = 0; i < tau_to_beta.size(); ++i) beta_path_by_start[tau_to_beta[i].sector(tau_to_beta[i].range.first_vertex)] = i;
    auto incomplete = [&](int sector) { return sector == -1 || propagator.get_zero_block_index(sector) == -1; };
    for (int mu = 0; mu < at_tau.size(); ++mu) {
      for (int i = 0; i < zero_to_tau.size(); ++i) {
        auto const &right = zero_to_tau[i];
        int right_end     = right.sector(right.range.vertex_end);
        if (incomplete(right_end)) continue;
        int left_start = at_tau[mu].get_block_index(right_end);
        if (incomplete(left_start)) continue;
        int j = beta_path_by_start[left_start];
        if (j == -1) continue;
        auto const &left = tau_to_beta[j];
        int left_end     = left.sector(left.range.vertex_end);
        if (incomplete(left_end)) continue;
        diagram::CorrelatorDiagram result{i, j, mu, {}};
        for (int kap = 0; kap < at_zero.size(); ++kap)
          if (at_zero[kap].get_block_index(left_end) == right.sector(right.range.first_vertex)) result.at_zero_operators.push_back(kap);
        visit(result);
      }
    }
  }
} // namespace triqs_xca::block_sparse

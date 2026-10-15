#include "correlator.hpp"

namespace triqs_xca::block_sparse::bind {
  CorrelatorOperands correlator(block_sparse::diagram::CorrelatorDiagram const &prepared,
                                std::span<block_sparse::diagram::BackbonePath const> zero_to_tau,
                                std::span<block_sparse::diagram::BackbonePath const> tau_to_beta, std::span<BlockOp const> at_tau,
                                std::span<BlockOp const> at_zero) {
    auto const &right = zero_to_tau[prepared.zero_to_tau_path];
    auto const &left  = tau_to_beta[prepared.tau_to_beta_path];
    CorrelatorOperands result{prepared.zero_to_tau_path,
                              prepared.tau_to_beta_path,
                              prepared.at_tau_operator,
                              at_tau[prepared.at_tau_operator].get_block(right.sector(right.vertex_end)),
                              {}};
    for (int kap : prepared.at_zero_operators) result.closures.push_back({kap, at_zero[kap].get_block(left.sector(left.vertex_end))});
    return result;
  }
} // namespace triqs_xca::block_sparse::bind

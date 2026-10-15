#pragma once
#include "triqs_xca/block_sparse/diagram/correlator.hpp"
#include "triqs_xca/block_sparse/correlator_operands.hpp"

namespace triqs_xca::block_sparse::bind {
CorrelatorOperands correlator(block_sparse::diagram::CorrelatorDiagram const &prepared,
    std::span<block_sparse::diagram::BackbonePath const> zero_to_tau,
    std::span<block_sparse::diagram::BackbonePath const> tau_to_beta,
    std::span<BlockOp const> at_tau, std::span<BlockOp const> at_zero);
} // namespace triqs_xca::block_sparse::bind

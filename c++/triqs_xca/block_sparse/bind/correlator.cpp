#include "correlator.hpp"

namespace triqs_xca::block_sparse::bind {
  CorrelatorOperands correlator(block_sparse::diagram::BackbonePath const &path, std::span<BlockOp const> at_tau, std::span<BlockOp const> at_zero) {
    path.backbone.require_correlator();
    if (path.range != path.backbone.range()) throw std::invalid_argument("correlator completion requires a complete backbone path");
    if (path.special_operators.size() != 1) throw std::invalid_argument("correlator completion requires one tau observable");
    int mu = path.special_operators.front();
    CorrelatorOperands result{mu, at_tau[mu].get_block(path.sector(path.backbone.endpoints.special_vertex)), {}};
    for (int kap : path.zero_operators) result.closures.push_back({kap, at_zero[kap].get_block(path.sector(0))});
    return result;
  }
} // namespace triqs_xca::block_sparse::bind

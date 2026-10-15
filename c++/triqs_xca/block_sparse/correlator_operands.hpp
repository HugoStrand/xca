#pragma once
#include <nda/nda.hpp>
#include <vector>

namespace triqs_xca::block_sparse {
struct CorrelatorClosureOperand {
  int at_zero_operator;
  nda::array_const_view<nda::dcomplex, 2> matrix;
};
/// External numerical operators for a prepared correlator completion. Backbone
/// results remain separate execution inputs, indexed in the caller's tables.
struct CorrelatorOperands {
  int zero_to_tau_path, tau_to_beta_path;
  int at_tau_operator;
  nda::array_const_view<nda::dcomplex, 2> at_tau_matrix;
  std::vector<CorrelatorClosureOperand> closures;
};
} // namespace triqs_xca::block_sparse

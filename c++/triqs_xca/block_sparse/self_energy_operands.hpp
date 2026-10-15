#pragma once
#include "backbone_operands.hpp"
#include "triqs_xca/diagram/sign.hpp"

namespace triqs_xca::block_sparse {
  struct EndpointOperand {
    nda::array_const_view<nda::dcomplex, 2> matrix;
    int coefficient_index;
    triqs_xca::diagram::Statistics statistics;
  };

  /// A factored endpoint sum: one coefficient view per late operand, with axes
  /// (time, early flavour). Backward orientation swaps the sliced coefficient
  /// index. There is no table of all endpoint pairs.
  struct EndpointSumOperands {
    std::vector<EndpointOperand> early;
    std::vector<EndpointOperand> late;
    std::vector<nda::array_const_view<nda::dcomplex, 2>> coefficients;
    int reflection_sign;
    triqs_xca::diagram::PermutationSigns signs;
  };

  /// Bound inputs for one self-energy contribution. Owns metadata, borrows values.
  struct SelfEnergyOperands {
    BackboneOperands backbone;
    EndpointSumOperands endpoints;
    int output_sector;
    int sign;
  };
  /// Reject nonclosing nonzero endpoint couplings before arithmetic. Exact-zero
  /// couplings return false; no cancellation between different diagrams is assumed.
  bool select_diagonal_self_energy(SelfEnergyOperands const &operands);
} // namespace triqs_xca::block_sparse

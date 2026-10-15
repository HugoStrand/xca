#pragma once
#include "triqs_xca/arc_data.hpp"

#include "self_energy_path.hpp"
#include "operator_binding.hpp"
#include "triqs_xca/diagram/sign.hpp"
#include "triqs_xca/hyb.hpp"

namespace triqs_xca::block_sparse {

struct EndpointOperand {
  nda::array_const_view<dcomplex, 2> matrix;
  int coefficient_index;
  triqs_xca::diagram::Statistics statistics;
};

/// A factored endpoint sum: one coefficient view per late operand, with axes
/// (time, early flavour). Backward orientation swaps the sliced coefficient
/// index. There is no table of all endpoint pairs.
struct EndpointOperands {
  std::vector<EndpointOperand> early;
  std::vector<EndpointOperand> late;
  std::vector<nda::array_const_view<dcomplex, 2>> coefficients;
  int reflection_sign;
  triqs_xca::diagram::PermutationSigns signs;
};

/// One path with resolved numerical operands. Metadata is owned; array views
/// borrow from the operator quartet, hybridization, and current propagator.
/// Those owners must remain alive and unchanged until execution finishes.
/// Execute with the matching assignment and mesh. This object must be rebound
/// after model/mesh/propagator replacement; it is
/// not a cache. Copies retain metadata independently of walker workspace reuse.
/// Storage is O(order + endpoint flavours), not O(number of paths or pairs).
struct SelfEnergyContribution {
  SelfEnergyPath path;
  std::vector<nda::array_const_view<dcomplex, 2>> vertices; // special endpoints have empty views
  std::vector<nda::array_const_view<dcomplex, 3>> propagators;
  EndpointOperands endpoints;
  int sign; // order, first-line direction, and scalar-kernel sign
};

/// Select a bound contribution for a block-diagonal output. A nonclosing path
/// with only exact-zero endpoint couplings contributes nothing and returns false.
/// Other nonclosing paths require a coarser partition and throw before execution.
/// This consumer requires closure per indexed contribution; cancellations between
/// different contributions are not used to establish a block-diagonal result.
bool select_diagonal_self_energy(SelfEnergyContribution const &contribution);

/// The path must have been discovered for this assignment and model.
SelfEnergyContribution bind_self_energy_contribution(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &assignment,
                                                     SelfEnergyPath const &path, BlockOpSymQuartet const &operators, OperatorSupports const &support,
                                                     triqs_xca::ArcData const &hybridization, std::span<triqs_xca::diagram::Statistics const> statistics,
                                                     triqs_xca::diagram::PermutationSigns signs);

} // namespace triqs_xca::block_sparse

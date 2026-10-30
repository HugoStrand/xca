#pragma once
#include "triqs_xca/block_sparse/self_energy_operands.hpp"
#include "backbone.hpp"

namespace triqs_xca::block_sparse::bind {
  EndpointSumOperands endpoint_sum(block_sparse::diagram::BackbonePath const &path, BlockOpSymQuartet const &operators,
                                   nda::array_const_view<nda::dcomplex, 3> tau_values, nda::array_const_view<nda::dcomplex, 3> tau_values_reflect,
                                   std::span<triqs_xca::Statistics const> statistics);
  SelfEnergyOperands self_energy(BlockDiagOpFun const &propagator, block_sparse::diagram::BackbonePath const &path,
                                 BlockOpSymQuartet const &operators, nda::array_const_view<nda::dcomplex, 3> tau_values,
                                 nda::array_const_view<nda::dcomplex, 3> tau_values_reflect, std::span<triqs_xca::Statistics const> statistics);
} // namespace triqs_xca::block_sparse::bind

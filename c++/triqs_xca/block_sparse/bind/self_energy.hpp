#pragma once
#include "triqs_xca/arc_data.hpp"
#include "triqs_xca/block_sparse/diagram/self_energy.hpp"
#include "triqs_xca/block_sparse/self_energy_operands.hpp"
#include "backbone.hpp"
#include "triqs_xca/block_sparse/operator_support.hpp"

namespace triqs_xca::block_sparse::bind {
  EndpointSumOperands endpoint_sum(triqs_xca::diagram::Diagram const &diagram, block_sparse::diagram::SelfEnergyDiagram const &prepared,
                                   BlockOpSymQuartet const &operators, OperatorSupportTable const &support, ArcData const &arc_data,
                                   std::span<triqs_xca::diagram::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs);
  SelfEnergyOperands self_energy(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram,
                                 block_sparse::diagram::SelfEnergyDiagram const &prepared, BlockOpSymQuartet const &operators,
                                 OperatorSupportTable const &support, ArcData const &arc_data,
                                 std::span<triqs_xca::diagram::Statistics const> statistics, triqs_xca::diagram::PermutationSigns signs);
} // namespace triqs_xca::block_sparse::bind

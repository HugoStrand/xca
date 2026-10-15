#pragma once
#include "triqs_xca/block_sparse/backbone_operands.hpp"
#include "vertex.hpp"

namespace triqs_xca::block_sparse::bind {
  BackboneOperands backbone(BlockDiagOpFun const &propagator, triqs_xca::diagram::Diagram const &diagram, BlockOpSymQuartet const &operators,
                            block_sparse::diagram::BackbonePath const &path);
} // namespace triqs_xca::block_sparse::bind

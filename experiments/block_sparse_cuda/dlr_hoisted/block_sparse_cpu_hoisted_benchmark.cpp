// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "hoisted_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
  return block_sparse_bench::run<triqs_xca::block_sparse::HoistedDiagramEvaluator>(
     argc, argv, "cpu_hoisted",
     [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<triqs_xca::block_sparse::HoistedDiagramEvaluator>(poles, coeff, mesh, ad, ops, dyn);
     },
     true);
}

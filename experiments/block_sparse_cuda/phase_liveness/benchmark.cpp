// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "lifetime_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<gpu::LifetimeDiagramEvaluator>(
     argc, argv, "cuda_phase_liveness", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<gpu::LifetimeDiagramEvaluator>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh);
     });
}

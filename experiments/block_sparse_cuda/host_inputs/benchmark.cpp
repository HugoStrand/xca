// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "host_input_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<gpu::HostInputDiagramEvaluator>(
     argc, argv, "cuda_host_inputs", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<gpu::HostInputDiagramEvaluator>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh);
     });
}

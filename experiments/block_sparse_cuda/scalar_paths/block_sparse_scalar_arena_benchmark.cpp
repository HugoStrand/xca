// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "scalar_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
#ifdef XCA_EXPERIMENT_SCALAR_CONSTANTS
  constexpr auto backend = "cuda_scalar_constants";
#else
  constexpr auto backend = "cuda_scalar_arena";
#endif
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<gpu::ScalarPathEvaluator>(
     argc, argv, backend, [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<gpu::ScalarPathEvaluator>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh, 1024ULL * 1024 * 1024,
                                                         32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
     });
}

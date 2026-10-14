// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "mapped_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<gpu::MappedKappaEvaluator>(
     argc, argv, "cuda_mapped_kappa", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       if (!gpu::ordered_sparse_available()) throw std::runtime_error("Ordered sparse kernels are unavailable");
       return std::make_unique<gpu::MappedKappaEvaluator>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh, 1024ULL * 1024 * 1024,
                                                          32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
     });
}

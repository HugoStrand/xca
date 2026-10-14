// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<gpu::DiagramEvaluator>(
     argc, argv, "cuda_mapped_production", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       if (!gpu::ordered_sparse_available() || !gpu::mapped_special_available()) throw std::runtime_error("Ordered/mapped kernels are unavailable");
       return std::make_unique<gpu::DiagramEvaluator>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh, 1ULL << 30, 32ULL << 20,
                                                      gpu::SparseBackend::Ordered, gpu::SpecialBackend::Mapped);
     });
}

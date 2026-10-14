// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kappa_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  std::string label = "cuda_kappa_" + std::to_string(XCA_EXPERIMENT_KAPPA_SLOTS);
  return block_sparse_bench::run<gpu::KappaCacheEvaluator>(
     argc, argv, label.c_str(), [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<gpu::KappaCacheEvaluator>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh, 1024ULL * 1024 * 1024,
                                                       32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
     });
}

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "row_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <cstdlib>
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<gpu::RowDiagramEvaluator>(
     argc, argv, "cuda_row_convolution", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       // Environment is experiment-only; invalid input rejects before evaluation.
       long rows = 8;
       if (auto *value = std::getenv("XCA_EXPERIMENT_CONV_ROWS")) {
         std::string text(value);
         std::size_t used = 0;
         rows             = std::stol(text, &used);
         if (used != text.size() || rows < 1 || rows > 512) throw std::invalid_argument("Invalid experiment row cap");
       }
       auto result = std::make_unique<gpu::RowDiagramEvaluator>(
          gpu::export_model(ad, coeff, ops, dyn), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Library,
          gpu::SpecialBackend::Reference, gpu::OutputPlacement::Resident, 512ULL * 1024 * 1024, gpu::NcaBackend::Reference, rows);
       if (std::getenv("XCA_EXPERIMENT_SAMPLE_ROWS")) result->sample_row_allocations(true);
       return result;
     });
}

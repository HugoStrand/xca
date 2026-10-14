// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
namespace triqs_xca::block_sparse::cuda {
  // Run the reusable observable suite with the explicit production policy.
  class HostOutputControl : public DiagramEvaluator {
    public:
    HostOutputControl(OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh,
                      std::size_t budget = 1024ULL * 1024 * 1024, std::size_t sparse_budget = 32ULL * 1024 * 1024,
                      SparseBackend sparse = SparseBackend::Library, SpecialBackend special = SpecialBackend::Reference)
       : DiagramEvaluator(std::move(model), poles, std::move(mesh), budget, sparse_budget, sparse, special, OutputPlacement::HostSectors) {}
  };
}

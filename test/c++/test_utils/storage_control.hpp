// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#ifdef NDA_HAVE_CUDA
namespace triqs_xca::block_sparse::cuda {
  template <StoragePolicy Policy> class StorageControl : public DiagramEvaluator {
    public:
    StorageControl(OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh, std::size_t budget = 1ULL << 30,
                   std::size_t sparse = 32ULL << 20, SparseBackend backend = SparseBackend::Library,
                   SpecialBackend special = SpecialBackend::Reference, OutputPlacement placement = OutputPlacement::Resident,
                   std::size_t host_budget = 512ULL << 20, NcaBackend nca = NcaBackend::Reference)
       : DiagramEvaluator(std::move(model), poles, std::move(mesh), budget, sparse, backend, special, placement, host_budget, nca, Policy) {}
  };
} // namespace triqs_xca::block_sparse::cuda
#endif

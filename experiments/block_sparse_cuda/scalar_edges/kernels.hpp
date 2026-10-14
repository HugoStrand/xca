// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace scalar_edge_experiment {
  cudaError_t ordered_fanin(void const *packed, std::uintptr_t const *outputs, long const *offsets, long const *lanes_list, long rank, long lanes,
                            long count, cudaStream_t stream);
  cudaError_t gather(std::uintptr_t const *f, std::uintptr_t const *g, void *fp, void *gp, long rank, long lanes, cudaStream_t stream);
  cudaError_t scatter(void const *packed, std::uintptr_t const *out, long rank, long lanes, cudaStream_t stream);
  cudaError_t multiply(void const *a, void const *b, void *out, long count, cudaStream_t stream);
  cudaError_t cross(void const *hf, void const *g, void const *f, void const *hg, void *out, long count, cudaStream_t stream);
} // namespace scalar_edge_experiment

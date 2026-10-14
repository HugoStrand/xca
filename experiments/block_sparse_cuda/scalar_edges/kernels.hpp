// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace scalar_edge_experiment {
  cudaError_t multiply(void const *a, void const *b, void *out, long count, cudaStream_t stream);
  cudaError_t cross(void const *hf, void const *g, void const *f, void const *hg, void *out, long count, cudaStream_t stream);
} // namespace scalar_edge_experiment

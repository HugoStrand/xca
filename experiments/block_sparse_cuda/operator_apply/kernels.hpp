// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace block_sparse_operator_experiment {
  // CSR rows own outputs; right applies the plain-transposed CSR snapshot.
  cudaError_t ordered_apply(long const *offsets, long const *columns, void const *values, void const *input, void *output, long rank, long rows,
                            long cols, long width, bool right, cudaStream_t stream);
} // namespace block_sparse_operator_experiment

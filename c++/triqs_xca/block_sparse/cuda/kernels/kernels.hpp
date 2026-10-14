// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
namespace triqs_xca::block_sparse::cuda::native {
  cudaError_t mapped_weighted_mu(void const *input, long const *sources, void const *values, void const *weights, void *output, long rank, long rows,
                                 long input_columns, long columns, long fields, cudaStream_t stream);
  // CSR rows own outputs; right applies the plain-transposed CSR snapshot.
  cudaError_t ordered_apply(long const *offsets, long const *columns, void const *values, void const *input, void *output, long rank, long rows,
                            long cols, long width, bool right, cudaStream_t stream);
} // namespace triqs_xca::block_sparse::cuda::native

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kernels.hpp"
#include <cuda_runtime.h>
namespace block_sparse_operator_experiment {
  template <bool Right>
  __global__ void apply(long const *offsets, long const *columns, double2 const *values, double2 const *input, double2 *output, long rank, long rows,
                        long cols, long width) {
    long index = long(blockIdx.x) * blockDim.x + threadIdx.x;
    long count = rank * rows * width;
    if (index >= count) return;
    long node = index / (rows * width), local = index % (rows * width);
    long row    = Right ? local % rows : local / width;
    long rhs    = Right ? local / rows : local % width;
    double2 sum = make_double2(0., 0.);
    for (long k = offsets[row]; k < offsets[row + 1]; ++k) {
      auto a = values[k];
      auto b = input[Right ? (node * width + rhs) * cols + columns[k] : (node * cols + columns[k]) * width + rhs];
      sum.x += a.x * b.x - a.y * b.y;
      sum.y += a.x * b.y + a.y * b.x;
    }
    output[index] = sum;
  }
  cudaError_t ordered_apply(long const *offsets, long const *columns, void const *values, void const *input, void *output, long rank, long rows,
                            long cols, long width, bool right, cudaStream_t stream) {
    auto count  = rank * rows * width;
    auto blocks = (count + 255) / 256;
    if (right)
      apply<true><<<blocks, 256, 0, stream>>>(offsets, columns, static_cast<double2 const *>(values), static_cast<double2 const *>(input),
                                              static_cast<double2 *>(output), rank, rows, cols, width);
    else
      apply<false><<<blocks, 256, 0, stream>>>(offsets, columns, static_cast<double2 const *>(values), static_cast<double2 const *>(input),
                                               static_cast<double2 *>(output), rank, rows, cols, width);
    return cudaGetLastError();
  }
} // namespace block_sparse_operator_experiment

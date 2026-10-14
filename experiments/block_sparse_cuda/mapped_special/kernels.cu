// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kernels.hpp"
#include <cuda_runtime.h>
namespace mapped_special_experiment {
  __device__ double2 product(double2 a, double2 b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
  __global__ void weighted_mu_kernel(double2 const *input, long const *sources, double2 const *values, double2 const *weights, double2 *output,
                                     long rank, long rows, long input_columns, long columns, long fields) {
    long index = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rank * rows * columns) return;
    long node = index / (rows * columns), row = (index / columns) % rows, column = index % columns;
    double2 sum{0., 0.};
    for (long field = 0; field < fields; ++field) {
      long map = field * columns + column, source = sources[map];
      double2 kappa{0., 0.};
      if (source >= 0) {
        auto term = product(values[map], input[(node * rows + row) * input_columns + source]);
        kappa.x += term.x;
        kappa.y += term.y;
      }
      auto weighted = product(kappa, weights[node * fields + field]);
      sum.x += weighted.x;
      sum.y += weighted.y;
    }
    output[index] = sum;
  }
  cudaError_t weighted_mu(void const *input, long const *sources, void const *values, void const *weights, void *output, long rank, long rows,
                          long input_columns, long columns, long fields, cudaStream_t stream) {
    long count = rank * rows * columns;
    weighted_mu_kernel<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<double2 const *>(input), sources, static_cast<double2 const *>(values),
                                                                static_cast<double2 const *>(weights), static_cast<double2 *>(output), rank, rows,
                                                                input_columns, columns, fields);
    return cudaGetLastError();
  }
} // namespace mapped_special_experiment

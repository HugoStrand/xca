// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kernels.hpp"
#include <cuda_runtime.h>
namespace scalar_edge_experiment {
  __device__ double2 product(double2 a, double2 b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
  __global__ void multiply_kernel(double2 const *a, double2 const *b, double2 *out, long count) {
    long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = product(a[i], b[i]);
  }
  __global__ void cross_kernel(double2 const *hf, double2 const *g, double2 const *f, double2 const *hg, double2 *out, long count) {
    long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
      auto a = product(hf[i], g[i]), b = product(f[i], hg[i]);
      out[i] = {a.x + b.x, a.y + b.y};
    }
  }
  cudaError_t multiply(void const *a, void const *b, void *out, long count, cudaStream_t stream) {
    multiply_kernel<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<double2 const *>(a), static_cast<double2 const *>(b),
                                                             static_cast<double2 *>(out), count);
    return cudaGetLastError();
  }
  cudaError_t cross(void const *hf, void const *g, void const *f, void const *hg, void *out, long count, cudaStream_t stream) {
    cross_kernel<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<double2 const *>(hf), static_cast<double2 const *>(g),
                                                          static_cast<double2 const *>(f), static_cast<double2 const *>(hg),
                                                          static_cast<double2 *>(out), count);
    return cudaGetLastError();
  }
} // namespace scalar_edge_experiment

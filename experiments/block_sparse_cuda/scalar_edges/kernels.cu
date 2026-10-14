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
  __global__ void gather_kernel(std::uintptr_t const *f, std::uintptr_t const *g, double2 *fp, double2 *gp, long rank, long lanes) {
    long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < rank * lanes) {
      long lane = i % lanes, node = i / lanes;
      fp[i] = reinterpret_cast<double2 const *>(f[lane])[node];
      gp[i] = reinterpret_cast<double2 const *>(g[lane])[node];
    }
  }
  __global__ void scatter_kernel(double2 const *packed, std::uintptr_t const *out, long rank, long lanes) {
    long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < rank * lanes) reinterpret_cast<double2 *>(out[i % lanes])[i / lanes] = packed[i];
  }
  cudaError_t gather(std::uintptr_t const *f, std::uintptr_t const *g, void *fp, void *gp, long rank, long lanes, cudaStream_t stream) {
    gather_kernel<<<(rank * lanes + 255) / 256, 256, 0, stream>>>(f, g, static_cast<double2 *>(fp), static_cast<double2 *>(gp), rank, lanes);
    return cudaGetLastError();
  }
  cudaError_t scatter(void const *packed, std::uintptr_t const *out, long rank, long lanes, cudaStream_t stream) {
    scatter_kernel<<<(rank * lanes + 255) / 256, 256, 0, stream>>>(static_cast<double2 const *>(packed), out, rank, lanes);
    return cudaGetLastError();
  }
  __global__ void fanin_kernel(double2 const *packed, std::uintptr_t const *outputs, long const *offsets, long const *lanes_list, long rank,
                               long lanes, long count) {
    long index = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rank * count) return;
    long node = index / count, slot = index % count;
    auto *destination = reinterpret_cast<double2 *>(outputs[slot]) + node;
    double2 sum       = *destination;
    // Stable per-output order, no atomics or tree reduction.
    for (long i = offsets[slot]; i < offsets[slot + 1]; ++i) {
      auto term = packed[node * lanes + lanes_list[i]];
      sum.x += term.x;
      sum.y += term.y;
    }
    *destination = sum;
  }
  cudaError_t ordered_fanin(void const *packed, std::uintptr_t const *outputs, long const *offsets, long const *lanes_list, long rank, long lanes,
                            long count, cudaStream_t stream) {
    fanin_kernel<<<(rank * count + 255) / 256, 256, 0, stream>>>(static_cast<double2 const *>(packed), outputs, offsets, lanes_list, rank, lanes,
                                                                 count);
    return cudaGetLastError();
  }
} // namespace scalar_edge_experiment

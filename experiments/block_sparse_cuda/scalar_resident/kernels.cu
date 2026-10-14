#include "resident_kernels.hpp"
#include <cuda_runtime.h>
namespace scalar_resident_experiment {
  __device__ double2 product(double2 a, double2 b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
  __global__ void apply_kernel(double2 *target, Weight const *weights, long rank, long lanes) {
    long index = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rank * lanes) return;
    auto descriptor = weights[index % lanes];
    double2 weight{double(descriptor.sign), 0.};
    if (descriptor.pointer) weight = product(weight, reinterpret_cast<double2 const *>(descriptor.pointer)[(index / lanes) * descriptor.stride]);
    target[index] = product(target[index], weight);
  }
  cudaError_t apply(void *target, Weight const *weights, long rank, long lanes, cudaStream_t stream) {
    apply_kernel<<<(rank * lanes + 255) / 256, 256, 0, stream>>>(static_cast<double2 *>(target), weights, rank, lanes);
    return cudaGetLastError();
  }
}

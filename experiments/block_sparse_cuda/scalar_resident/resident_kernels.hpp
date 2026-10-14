#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace scalar_resident_experiment {
  struct Weight {
    std::uintptr_t pointer = 0;
    long stride = 1;
    int sign = 1;
  };
  cudaError_t apply(void *target, Weight const *weights, long rank, long lanes, cudaStream_t stream);
}

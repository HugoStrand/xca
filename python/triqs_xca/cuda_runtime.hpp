#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
namespace xca_python {
inline void check_cuda_runtime(cudaError_t status) {
  if(status != cudaSuccess) throw std::runtime_error(std::string("CUDA runtime: ")+cudaGetErrorString(status));
}
inline int cuda_device_count() {
  int count=0; check_cuda_runtime(cudaGetDeviceCount(&count)); return count;
}
inline int cuda_current_device() {
  int device=0; check_cuda_runtime(cudaGetDevice(&device)); return device;
}
inline void cuda_select_device(int device) {
  if(device<0 || device>=cuda_device_count()) throw std::out_of_range("CUDA device ordinal is unavailable");
  check_cuda_runtime(cudaSetDevice(device));
}
}

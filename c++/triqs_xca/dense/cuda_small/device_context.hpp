// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef XCA_HAVE_CUDA_SMALL
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace triqs_xca::dense::cuda_small {
  // Declare first in each owner so it outlives all device buffers/handles.
  // Evaluation requires the construction device. Destruction temporarily selects
  // that device, then restores the caller's selection after member cleanup.
  class DeviceContext {
    int device_ = -1, restore_ = -1;
    static void check(cudaError_t status) {
      if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA-small device context: ") + cudaGetErrorString(status));
    }

    public:
    DeviceContext() { check(cudaGetDevice(&device_)); }
    DeviceContext(DeviceContext const &)            = delete;
    DeviceContext &operator=(DeviceContext const &) = delete;
    int device() const { return device_; }
    void require_current() const {
      int current = -1;
      check(cudaGetDevice(&current));
      if (current != device_) throw std::runtime_error("CUDA-small evaluator requires its construction device");
    }
    void begin_cleanup() noexcept {
      int current = -1;
      if (cudaGetDevice(&current) == cudaSuccess && current != device_ && cudaSetDevice(device_) == cudaSuccess) restore_ = current;
    }
    ~DeviceContext() {
      if (restore_ >= 0) cudaSetDevice(restore_);
    }
  };
} // namespace triqs_xca::dense::cuda_small
#endif

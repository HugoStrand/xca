// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef NDA_HAVE_CUDA
#include <nvtx3/nvToolsExt.h>
namespace triqs_xca::dense::cuda {
  // Header-only NVTX dispatch is inert without an attached profiler. A range
  // surrounds host scheduling as well as the default-stream work it enqueues.
  struct ProfileRange {
    explicit ProfileRange(char const *name) { nvtxRangePushA(name); }
    ~ProfileRange() { nvtxRangePop(); }
    ProfileRange(ProfileRange const &)            = delete;
    ProfileRange &operator=(ProfileRange const &) = delete;
  };
} // namespace triqs_xca::dense::cuda
#endif

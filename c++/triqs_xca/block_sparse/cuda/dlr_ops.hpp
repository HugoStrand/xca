// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "device_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include <cppdlr/dlr_imtime.hpp>

namespace triqs_xca::block_sparse::cuda {
  // Immutable rank tables/LU snapshot, own cuSOLVER handle on the shared stream.
  // Scratch follows actual rectangular shapes. Budget covers explicit tables,
  // pivots/status, RHS and convolution scratch, excluding caller input/output and
  // opaque library allocations. One owner at a time; public calls are synchronous.
  class DlrOps {
    struct Impl;
    std::unique_ptr<Impl> impl_;

    public:
    DlrOps(std::shared_ptr<DeviceOps> ops, cppdlr::imtime_ops const &host, std::size_t storage_budget = 256ULL * 1024 * 1024);
    ~DlrOps();
    DlrOps(DlrOps const &)            = delete;
    DlrOps &operator=(DlrOps const &) = delete;
    long rank() const;
    std::size_t storage_bytes() const;
    void vals2coefs(TensorConstView values, TensorView coefficients);
    void coefs2vals(TensorConstView coefficients, TensorView values) const;
    void reflect(TensorConstView values, TensorView reflected) const;
    // Coefficient inputs, time-ordered value output; out may alias an input when
    // their shapes match. Input reads finish before any caller output is written.
    void convolve(double beta, TensorConstView fc, TensorConstView gc, TensorView out);
  };
} // namespace triqs_xca::block_sparse::cuda
#endif

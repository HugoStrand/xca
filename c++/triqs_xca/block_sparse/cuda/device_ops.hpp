// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sparse_operator.hpp"
#ifdef NDA_HAVE_CUDA
#include <nda/device.hpp>
#include <memory>

namespace triqs_xca::block_sparse::cuda {
  using Tensor          = nda::cuarray<nda::dcomplex, 3>;
  using TensorView      = nda::cuarray_view<nda::dcomplex, 3>;
  using TensorConstView = nda::cuarray_const_view<nda::dcomplex, 3>;

  // Shared by one evaluator's operators/transforms. Owns a nonblocking stream and
  // library handles; never borrows NDA's process-global handles. Calls complete
  // synchronously at the public boundary. Inputs must be contiguous C-order.
  class DeviceOps {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class DeviceSparseBlock;

    public:
    DeviceOps();
    ~DeviceOps();
    DeviceOps(DeviceOps const &)            = delete;
    DeviceOps &operator=(DeviceOps const &) = delete;
    void require_current() const;
    int device() const;
    cudaStream_t stream() const;
    void finish() const;
    // Explicit event handoff for NDA uploads on the default stream.
    void wait_for_default_uploads() const;
    void zero(TensorView out) const;
    void copy(TensorConstView in, TensorView out) const;
    void axpy(nda::dcomplex alpha, TensorConstView in, TensorView out) const;
    // Per-node (M,K)*(K,N), with arbitrary rectangular spatial extents.
    void product(TensorConstView left, TensorConstView right, TensorView out) const;
    // Apply a contiguous (r,r) rank matrix to a rectangular (r,M,N) function.
    void rank_product(nda::cumatrix_const_view<nda::dcomplex> op, TensorConstView in, TensorView out) const;
  };

  // An owned immutable CSR snapshot, including its plain transpose for right
  // products. No dense expansion or adjoint substitution. Deterministic ALG3 is
  // used one DLR node at a time. Workspace is retained/reused within the budget.
  class DeviceSparseBlock {
    struct Impl;
    std::unique_ptr<Impl> impl_;

    public:
    DeviceSparseBlock(std::shared_ptr<DeviceOps> ops, SparseBlock const &host, std::size_t storage_budget = 32ULL * 1024 * 1024);
    ~DeviceSparseBlock();
    DeviceSparseBlock(DeviceSparseBlock const &)            = delete;
    DeviceSparseBlock &operator=(DeviceSparseBlock const &) = delete;
    long rows() const;
    long columns() const;
    std::size_t storage_bytes() const;
    std::size_t workspace_bytes() const;
    void apply_left(TensorConstView in, TensorView out);
    void apply_right(TensorConstView in, TensorView out);
  };
} // namespace triqs_xca::block_sparse::cuda
#endif

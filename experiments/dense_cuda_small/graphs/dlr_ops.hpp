// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "device_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include <cppdlr/dlr_imtime.hpp>
#include <cusolverDn.h>

namespace xca_experiments::graphs {

  // Device counterpart of the DLR operations used by the dense evaluator.
  // Tables are immutable constructor snapshots. Scratch is reused, not thread-safe.
  // Owns solver handle and one persistent status slot per enqueued solve.
  // Status is inspected only after stream completion; no shared NDA settings.
  class DlrOps {
    int r_, N_, columns_;
    DeviceOps ops_;
    nda::cumatrix<complex> lu_, C_, H_, Q_, reflection_;
    nda::cuvector<int> pivots_;
    nda::cumatrix<complex, nda::F_layout> rhs_;
    tensor hf_, hg_, product_, cross_;
    cusolverDnHandle_t solver_ = nullptr;
    nda::vector<int, nda::heap<nda::mem::Unified>> status_;
    int used_status_ = 0;
    void validate(tensor_const_view input) const;

    public:
    DlrOps(cppdlr::imtime_ops const &host, long dimension);
    ~DlrOps();
    void begin() { used_status_ = 0; }
    int status_count() const { return used_status_; }
    void set_status_count(int n) { used_status_ = n; }
    void finish();
    void wait_for_default_uploads() { ops_.wait_for_default_uploads(); }
    DlrOps(DlrOps const &)            = delete;
    DlrOps &operator=(DlrOps const &) = delete;
    std::size_t workspace_bytes() const;
    DeviceOps &device_ops() { return ops_; }
    // In-place values->coefficients is safe because the solve uses owned RHS storage.
    void vals2coefs(tensor_const_view values, tensor_view coefficients);
    // Rank GEMMs require disjoint input/output, as for other device products.
    void coefs2vals(tensor_const_view coefficients, tensor_view values) const;
    void reflect(tensor_const_view values, tensor_view reflected) const;
    // Matrix-valued TIME_ORDERED convolution, with coefficient inputs/value output.
    // out may alias either input; all input reads finish before the final transforms.
    void convolve(double beta, tensor_const_view fc, tensor_const_view gc, tensor_view out);
  };

} // namespace xca_experiments::graphs
#endif

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "device_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include <cppdlr/dlr_imtime.hpp>

namespace triqs_xca::dense::cuda {

  // Device counterpart of the DLR operations used by the dense evaluator.
  // Tables are immutable constructor snapshots. Scratch is reused, not thread-safe.
  // NDA uses process-shared solver handle/status; serialize device solver calls
  // with other NDA users as well. Solver synchronization must remain enabled.
  class DlrOps {
    int r_, N_, columns_;
    DeviceOps ops_;
    nda::cumatrix<complex> lu_, C_, H_, Q_, reflection_;
    nda::cuvector<int> pivots_;
    nda::cumatrix<complex, nda::F_layout> rhs_;
    tensor hf_, hg_, product_, cross_;
    void validate(tensor_const_view input) const;

    public:
    DlrOps(cppdlr::imtime_ops const &host, long dimension);
    DlrOps(DlrOps const &)            = delete;
    DlrOps &operator=(DlrOps const &) = delete;
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

} // namespace triqs_xca::dense::cuda
#endif

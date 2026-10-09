// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nda/nda.hpp>

#ifdef NDA_HAVE_CUDA
#include <nda/device.hpp>
#include <cstddef>

namespace triqs_xca::dense::cuda {

  using complex            = nda::dcomplex;
  using tensor             = nda::cuarray<complex, 3>;
  using tensor_view        = nda::cuarray_view<complex, 3>;
  using tensor_const_view  = nda::cuarray_const_view<complex, 3>;
  using matrix_view        = nda::cumatrix_view<complex>;
  using matrix_const_view  = nda::cumatrix_const_view<complex>;
  using fmatrix_view       = nda::cumatrix_view<complex, nda::F_layout>;
  using fmatrix_const_view = nda::cumatrix_const_view<complex, nda::F_layout>;

  // Contiguous C-order (r,N,N) primitives. Input/output products never alias.
  // The default stream and synchronous completion match NDA's initial semantics.
  // Pointer tables are allocated once and reused; this object is not thread-safe.
  class DeviceOps {
    int r_, N_, stride_, count_;
    nda::vector<complex const *, nda::heap<nda::mem::Unified>> a_ptrs_, b_ptrs_;
    nda::vector<complex *, nda::heap<nda::mem::Unified>> c_ptrs_;
    cublasHandle_t handle_ = nullptr;
    void validate(tensor_const_view a) const;
    void validate_operator(matrix_const_view a) const;
    void vertex(matrix_const_view op, tensor_const_view in, tensor_view out, bool left, complex alpha, complex beta);

    public:
    static std::size_t checked_product(std::size_t a, std::size_t b);
    static int checked_blas_int(long value);
    DeviceOps(long rank, long dimension);
    ~DeviceOps();
    DeviceOps(DeviceOps const &)            = delete;
    DeviceOps &operator=(DeviceOps const &) = delete;

    void zero(tensor_view out) const;
    void copy(tensor_const_view in, tensor_view out) const;
    void scale(complex alpha, tensor_view out) const;
    void axpy(complex alpha, tensor_const_view in, tensor_view out) const;
    void batch_product(tensor_const_view a, tensor_const_view b, tensor_view out, complex alpha = 1., complex beta = 0.) const;
    // Apply a contiguous (r,r) matrix along the rank axis of (r,N,N).
    void rank_product(matrix_const_view op, tensor_const_view in, tensor_view out, complex alpha = 1., complex beta = 0.) const;
    void left_product(matrix_const_view op, tensor_const_view in, tensor_view out, complex alpha = 1., complex beta = 0.);
    void right_product(tensor_const_view in, matrix_const_view op, tensor_view out, complex alpha = 1., complex beta = 0.);
    // Flatten (r,N,N) as a C-order (r,N*N) matrix. Weights are complex device scalars.
    void weight_rows(tensor_const_view in, nda::cuvector_const_view<complex> weights, tensor_view out) const;
    // Logical matrix values are unchanged; GEAM transposes only physical storage.
    void to_fortran(matrix_const_view in, fmatrix_view out) const;
    void from_fortran(fmatrix_const_view in, matrix_view out) const;
    // packed[k,i*N+j] = operator_k(j,i).
    // out[t,k] = sum_ij in[t,i,j] * packed[k,i*N+j], without conjugation.
    void traces(tensor_const_view in, matrix_const_view packed, matrix_view out) const;
  };

} // namespace triqs_xca::dense::cuda
#endif

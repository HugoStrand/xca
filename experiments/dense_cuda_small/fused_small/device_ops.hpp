// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nda/nda.hpp>

#ifdef NDA_HAVE_CUDA
#include <cstddef>
#include <map>
#include <memory>
#include <nda/device.hpp>
#include <tuple>

namespace xca_experiments::fused_small {

  using complex            = nda::dcomplex;
  using tensor             = nda::cuarray<complex, 3>;
  using tensor_view        = nda::cuarray_view<complex, 3>;
  using tensor_const_view  = nda::cuarray_const_view<complex, 3>;
  using matrix_view        = nda::cumatrix_view<complex>;
  using matrix_const_view  = nda::cumatrix_const_view<complex>;
  using fmatrix_view       = nda::cumatrix_view<complex, nda::F_layout>;
  using fmatrix_const_view = nda::cumatrix_const_view<complex, nda::F_layout>;

  // Contiguous (r,B,N,N), exposed as (r*B,N,N). Rank transforms share r-by-r
  // tables. Input/output products never alias. Owned nonblocking stream;
  // operations enqueue and finish() completes them. Immutable pointer tables are
  // prepared before traversal; not thread-safe.
  class DeviceOps {
    int r_, N_, stride_, count_, rank_, B_;
    int mode_ = 1;
    using Key = std::tuple<complex const *, complex const *, complex *, bool>;
    struct Tables {
      nda::cuvector<complex const *> a, b;
      nda::cuvector<complex *> c;
      explicit Tables(int r) : a(r), b(r), c(r) {}
    };
    std::map<Key, std::unique_ptr<Tables>> tables_;
    cublasHandle_t handle_    = nullptr;
    cudaStream_t stream_      = nullptr;
    cudaEvent_t upload_ready_ = nullptr;
    void validate(tensor_const_view a) const;
    void validate_operator(matrix_const_view a) const;
    void vertex(matrix_const_view op, tensor_const_view in, tensor_view out, bool left, complex alpha, complex beta);

    public:
    static std::size_t checked_sum(std::size_t a, std::size_t b);
    std::size_t workspace_bytes() const;
    static std::size_t checked_product(std::size_t a, std::size_t b);
    static int checked_blas_int(long value);
    DeviceOps(long rank, long dimension, int capacity);
    ~DeviceOps();
    DeviceOps(DeviceOps const &)            = delete;
    DeviceOps &operator=(DeviceOps const &) = delete;

    struct LaneTables {
      nda::cuvector<complex const *> a, b;
      nda::cuvector<complex *> c;
      explicit LaneTables(int n) : a(n), b(n), c(n) {}
    };
    void prepare_lanes(LaneTables &table, std::vector<complex const *> const &operators, tensor_const_view in, tensor_view out, bool left);
    void lane_vertex(LaneTables const &table, complex alpha = 1., complex beta = 0.) const;
    void broadcast(tensor_const_view input, tensor_view out, int active) const;
    void zero_output(tensor_view out) const;
    void accumulate_lane(tensor_const_view input, tensor_view scratch, tensor_view sum, int lane, complex alpha) const;
    cudaStream_t stream() const { return stream_; }
    void finish() const;
    int mode() const { return mode_; }
    void weighted_accumulate(tensor_const_view in, nda::cuvector_const_view<complex> w, nda::cuvector_const_view<complex> parity,
                             tensor_view out) const;
    void axpy_scale(tensor_const_view in, tensor_view out, double scale) const;
    // Call after external NDA/default-stream uploads, before owned-stream reads.
    void wait_for_default_uploads() const;
    void prepare_vertex(matrix_const_view op, tensor_const_view in, tensor_view out, bool left);
    void zero(tensor_view out) const;
    void copy(tensor_const_view in, tensor_view out) const;
    void scale(complex alpha, tensor_view out) const;
    void axpy(complex alpha, tensor_const_view in, tensor_view out) const;
    void batch_product(tensor_const_view a, tensor_const_view b, tensor_view out, complex alpha = 1., complex beta = 0.) const;
    // Apply a contiguous (r,r) matrix along the rank axis of (r,N,N).
    void rank_product(matrix_const_view op, tensor_const_view in, tensor_view out, complex alpha = 1., complex beta = 0.) const;
    void left_product(matrix_const_view op, tensor_const_view in, tensor_view out, complex alpha = 1., complex beta = 0.);
    void right_product(tensor_const_view in, matrix_const_view op, tensor_view out, complex alpha = 1., complex beta = 0.);
    // Flatten (r,N,N) as a C-order (r,N*N) matrix. Weights are complex device
    // scalars.
    void weight_rows(tensor_const_view in, nda::cuvector_const_view<complex> weights, tensor_view out) const;
    // Logical matrix values are unchanged; GEAM transposes only physical storage.
    void to_fortran(matrix_const_view in, fmatrix_view out) const;
    void from_fortran(fmatrix_const_view in, matrix_view out) const;
    // packed[k,i*N+j] = operator_k(j,i).
    // out[t,k] = sum_ij in[t,i,j] * packed[k,i*N+j], without conjugation.
    void traces(tensor_const_view in, matrix_const_view packed, matrix_view out) const;
  };

} // namespace xca_experiments::fused_small
#endif

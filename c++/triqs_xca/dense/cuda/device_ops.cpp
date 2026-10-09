// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "device_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include <nda/blas.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace triqs_xca::dense::cuda {
  namespace {
    void check(cublasStatus_t status, char const *operation) {
      if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("dense CUDA ") + operation + ": cuBLAS status " + std::to_string(status));
    }
    void finish(char const *operation) { nda::cuda_device_sync(true, operation); }
    void require(bool condition, char const *message) {
      if (!condition) throw std::invalid_argument(std::string("dense CUDA: ") + message);
    }
    void disjoint(complex const *a, std::size_t na, complex const *b, std::size_t nb) {
      auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
      auto nx = DeviceOps::checked_product(na, sizeof(complex)), ny = DeviceOps::checked_product(nb, sizeof(complex));
      require(x <= std::numeric_limits<std::uintptr_t>::max() - nx && y <= std::numeric_limits<std::uintptr_t>::max() - ny, "address range overflow");
      require(x + nx <= y || y + ny <= x, "input and output must not overlap");
    }
  } // namespace

  std::size_t DeviceOps::checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("dense CUDA: allocation/stride overflow");
    return a * b;
  }
  int DeviceOps::checked_blas_int(long value) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) throw std::invalid_argument("dense CUDA: dimension exceeds BLAS range");
    return static_cast<int>(value);
  }
  DeviceOps::DeviceOps(long rank, long dimension)
     : r_(checked_blas_int(rank)),
       N_(checked_blas_int(dimension)),
       stride_(checked_blas_int(checked_product(N_, N_))),
       count_(checked_blas_int(checked_product(r_, stride_))),
       a_ptrs_(r_),
       b_ptrs_(r_),
       c_ptrs_(r_) {
    checked_product(count_, sizeof(complex));
    check(cublasCreate(&handle_), "create handle");
    try {
      check(cublasSetStream(handle_, nullptr), "set default stream");
      check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "set scalar pointer mode");
    } catch (...) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }
  DeviceOps::~DeviceOps() {
    if (handle_) cublasDestroy(handle_);
  }
  void DeviceOps::validate(tensor_const_view a) const {
    require(a.shape() == std::array<long, 3>{r_, N_, N_}, "wrong tensor shape");
    require(a.strides() == std::array<long, 3>{stride_, N_, 1}, "tensor must be contiguous C-order");
  }
  void DeviceOps::validate_operator(matrix_const_view a) const {
    require(a.shape() == std::array<long, 2>{N_, N_}, "wrong operator shape");
    require(a.strides() == std::array<long, 2>{N_, 1}, "operator must be contiguous C-order");
  }
  void DeviceOps::zero(tensor_view out) const {
    validate(out);
    nda::device_error_check(cudaMemsetAsync(out.data(), 0, checked_product(count_, sizeof(complex)), nullptr), "dense CUDA zero");
    finish("dense CUDA zero");
  }
  void DeviceOps::copy(tensor_const_view in, tensor_view out) const {
    validate(in);
    validate(out);
    if (in.data() == out.data()) return;
    disjoint(in.data(), count_, out.data(), count_);
    nda::device_error_check(cudaMemcpyAsync(out.data(), in.data(), checked_product(count_, sizeof(complex)), cudaMemcpyDeviceToDevice, nullptr),
                            "dense CUDA copy");
    finish("dense CUDA copy");
  }
  void DeviceOps::scale(complex alpha, tensor_view out) const {
    validate(out);
    nda::blas::device::scal(count_, alpha, out.data(), 1);
    finish("dense CUDA scale");
  }
  void DeviceOps::axpy(complex alpha, tensor_const_view in, tensor_view out) const {
    validate(in);
    validate(out);
    disjoint(in.data(), count_, out.data(), count_);
    nda::blas::device::axpy(count_, alpha, in.data(), 1, out.data(), 1);
    finish("dense CUDA axpy");
  }
  void DeviceOps::batch_product(tensor_const_view a, tensor_const_view b, tensor_view out, complex alpha, complex beta) const {
    validate(a);
    validate(b);
    validate(out);
    disjoint(a.data(), count_, out.data(), count_);
    disjoint(b.data(), count_, out.data(), count_);
    // Row-major C=A*B becomes column-major C^T=B^T*A^T; T is not conjugation.
    nda::blas::device::gemm_batch_strided('N', 'N', N_, N_, N_, alpha, b.data(), N_, stride_, a.data(), N_, stride_, beta, out.data(), N_, stride_,
                                          r_);
    finish("dense CUDA batched product");
  }
  void DeviceOps::vertex(matrix_const_view op, tensor_const_view in, tensor_view out, bool left, complex alpha, complex beta) {
    validate_operator(op);
    validate(in);
    validate(out);
    disjoint(op.data(), stride_, out.data(), count_);
    disjoint(in.data(), count_, out.data(), count_);
    for (int t = 0; t < r_; ++t) {
      a_ptrs_(t) = left ? in.data() + std::size_t(t) * stride_ : op.data();
      b_ptrs_(t) = left ? op.data() : in.data() + std::size_t(t) * stride_;
      c_ptrs_(t) = out.data() + std::size_t(t) * stride_;
    }
    nda::blas::device::gemm_batch('N', 'N', N_, N_, N_, alpha, a_ptrs_.data(), N_, b_ptrs_.data(), N_, beta, c_ptrs_.data(), N_, r_);
    finish("dense CUDA vertex product");
  }
  void DeviceOps::left_product(matrix_const_view op, tensor_const_view in, tensor_view out, complex alpha, complex beta) {
    vertex(op, in, out, true, alpha, beta);
  }
  void DeviceOps::right_product(tensor_const_view in, matrix_const_view op, tensor_view out, complex alpha, complex beta) {
    vertex(op, in, out, false, alpha, beta);
  }
  void DeviceOps::weight_rows(tensor_const_view in, nda::cuvector_const_view<complex> weights, tensor_view out) const {
    validate(in);
    validate(out);
    require(weights.size() == r_ && weights.strides()[0] == 1, "wrong/non-contiguous time weights");
    disjoint(in.data(), count_, out.data(), count_);
    disjoint(weights.data(), r_, out.data(), count_);
    // C-order r x N^2 is physically column-major N^2 x r; scale its columns.
    check(cublasZdgmm(handle_, CUBLAS_SIDE_RIGHT, stride_, r_, nda::cucplx(in.data()), stride_, nda::cucplx(weights.data()), 1,
                      nda::cucplx(out.data()), stride_),
          "time-row scaling");
    finish("dense CUDA time-row scaling");
  }
  void DeviceOps::to_fortran(matrix_const_view in, fmatrix_view out) const {
    require(in.shape() == out.shape(), "layout conversion shape mismatch");
    int rows = checked_blas_int(in.extent(0)), cols = checked_blas_int(in.extent(1));
    require(in.strides() == std::array<long, 2>{cols, 1} && out.strides() == std::array<long, 2>{1, rows}, "invalid conversion layout");
    auto size = checked_product(rows, cols);
    disjoint(in.data(), size, out.data(), size);
    auto one = nda::cucplx(complex{1.}), zero_value = nda::cucplx(complex{0.});
    check(cublasZgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T, rows, cols, &one, nda::cucplx(in.data()), cols, &zero_value, nda::cucplx(in.data()), cols,
                      nda::cucplx(out.data()), rows),
          "C-to-F pack");
    finish("dense CUDA C-to-F pack");
  }
  void DeviceOps::from_fortran(fmatrix_const_view in, matrix_view out) const {
    require(in.shape() == out.shape(), "layout conversion shape mismatch");
    int rows = checked_blas_int(in.extent(0)), cols = checked_blas_int(in.extent(1));
    require(in.strides() == std::array<long, 2>{1, rows} && out.strides() == std::array<long, 2>{cols, 1}, "invalid conversion layout");
    auto size = checked_product(rows, cols);
    disjoint(in.data(), size, out.data(), size);
    auto one = nda::cucplx(complex{1.}), zero_value = nda::cucplx(complex{0.});
    check(cublasZgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_T, cols, rows, &one, nda::cucplx(in.data()), rows, &zero_value, nda::cucplx(in.data()), rows,
                      nda::cucplx(out.data()), cols),
          "F-to-C unpack");
    finish("dense CUDA F-to-C unpack");
  }
  void DeviceOps::traces(tensor_const_view in, matrix_const_view packed, matrix_view out) const {
    validate(in);
    int k = checked_blas_int(packed.extent(0));
    require(packed.extent(1) == stride_ && packed.strides() == std::array<long, 2>{stride_, 1}, "invalid packed trace operators");
    require(out.shape() == std::array<long, 2>{r_, k} && out.strides() == std::array<long, 2>{k, 1}, "invalid trace output");
    disjoint(in.data(), count_, out.data(), checked_product(r_, k));
    disjoint(packed.data(), checked_product(k, stride_), out.data(), checked_product(r_, k));
    // Y=A*packed^T: reinterpret C-order arrays in F order and transpose without conjugation.
    nda::blas::device::gemm('T', 'N', k, r_, stride_, complex{1.}, packed.data(), stride_, in.data(), stride_, complex{0.}, out.data(), k);
    finish("dense CUDA trace contraction");
  }
} // namespace triqs_xca::dense::cuda
#endif

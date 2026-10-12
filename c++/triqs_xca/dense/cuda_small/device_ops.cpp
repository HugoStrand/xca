// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "device_ops.hpp"
#include <cstdlib>
#ifdef XCA_HAVE_CUDA_SMALL
#include "kernels.hpp"
#include "qualification.hpp"
#include <cusolver_common.h>
#include <cstdint>
#include <limits>
#include <nda/blas.hpp>
#include <stdexcept>
#include <string>

namespace triqs_xca::dense::cuda_small {
  namespace {
    void check(cublasStatus_t status, char const *operation) {
      if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("dense CUDA ") + operation + ": cuBLAS status " + std::to_string(status));
    }

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

  std::size_t DeviceOps::checked_sum(std::size_t a, std::size_t b) {
    if (a > SIZE_MAX - b) throw std::overflow_error("dense CUDA workspace size overflow");
    return a + b;
  }
  std::size_t DeviceOps::table_host_bound_bytes() {
    // Map payload/node links plus the separately allocated Tables object.
    return checked_sum(sizeof(std::pair<Key const, std::unique_ptr<Tables>>) + 4 * sizeof(void *), sizeof(Tables));
  }
  std::size_t DeviceOps::table_entry_bytes() const {
    return checked_sum(checked_product(3, checked_product(r_, sizeof(complex *))), table_host_bound_bytes());
  }
  std::size_t DeviceOps::device_workspace_bytes() const {
    return checked_product(tables_.size(), checked_product(3, checked_product(r_, sizeof(complex *))));
  }
  std::size_t DeviceOps::host_workspace_bound_bytes() const { return checked_product(tables_.size(), table_host_bound_bytes()); }
  std::size_t DeviceOps::workspace_bytes() const { return checked_sum(device_workspace_bytes(), host_workspace_bound_bytes()); }
  std::size_t DeviceOps::checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("dense CUDA: allocation/stride overflow");
    return a * b;
  }
  int DeviceOps::checked_blas_int(long value) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) throw std::invalid_argument("dense CUDA: dimension exceeds BLAS range");
    return static_cast<int>(value);
  }
  DeviceOps::DeviceOps(long rank, long dimension, int capacity)
     : r_(checked_blas_int(checked_product(rank, capacity))),
       N_(checked_blas_int(dimension)),
       stride_(checked_blas_int(checked_product(N_, N_))),
       count_(checked_blas_int(checked_product(r_, stride_))),
       rank_(checked_blas_int(rank)),
       B_(capacity) {
    checked_product(count_, sizeof(complex));
    // Conservative initial eligibility; expand after method/rank qualification.
    mode_ = (rank == 16 && (N_ == 4 || N_ == 8)) ? 2 : 0;
    nda::device_error_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create owned stream");
    try {
      nda::device_error_check(cudaEventCreateWithFlags(&upload_ready_, cudaEventDisableTiming), "create upload event");
      check(cublasCreate(&handle_), "create handle");
      check(cublasSetStream(handle_, stream_), "set default stream");
      check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "set scalar pointer mode");
      cudaDeviceProp property{};
      int runtime = 0, driver = 0;
      nda::device_error_check(cudaGetDeviceProperties(&property, context_.device()), "CUDA-small device properties");
      nda::device_error_check(cudaRuntimeGetVersion(&runtime), "CUDA-small runtime version");
      nda::device_error_check(cudaDriverGetVersion(&driver), "CUDA-small driver version");
      std::array<int, 3> blas{}, solver{};
      for (int i = 0; i < 3; ++i) {
        check(cublasGetProperty(static_cast<libraryPropertyType>(i), &blas[i]), "cuBLAS library version");
        if (cusolverGetProperty(static_cast<libraryPropertyType>(i), &solver[i]) != CUSOLVER_STATUS_SUCCESS)
          throw std::runtime_error("CUDA-small cuSOLVER library version failed");
      }
      QualificationEnvironment environment{property.name, property.major, property.minor, CUDART_VERSION, runtime, driver, blas, solver};
      if (!qualified_environment(environment)) mode_ = 0;
    } catch (...) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      if (upload_ready_) cudaEventDestroy(upload_ready_);
      upload_ready_ = nullptr;
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
      throw;
    }
  }
  DeviceOps::~DeviceOps() {
    context_.begin_cleanup();
    if (stream_) cudaStreamSynchronize(stream_);
    if (handle_) cublasDestroy(handle_);
    if (upload_ready_) cudaEventDestroy(upload_ready_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  void DeviceOps::wait_for_default_uploads() const {
    context_.require_current();
    nda::device_error_check(cudaEventRecord(upload_ready_, nullptr), "record default uploads");
    nda::device_error_check(cudaStreamWaitEvent(stream_, upload_ready_, 0), "wait for default uploads");
  }
  void DeviceOps::finish() const {
    context_.require_current();
    // Successful draining must not allocate an error-message string.
    auto status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) nda::device_error_check(status, "serial_stream completion");
  }
  void DeviceOps::prepare_vertex(matrix_const_view op, tensor_const_view in, tensor_view out, bool left) {
    Key key{op.data(), in.data(), out.data(), left};
    if (tables_.contains(key)) return;
    auto bytes = table_entry_bytes();
    if (bytes > table_budget_) throw std::length_error("CUDA-small vertex table exceeds workspace budget");
    if (checked_sum(workspace_bytes(), bytes) > table_budget_) clear_tables();
    auto table = std::make_unique<Tables>(r_);
    nda::vector<complex const *> ap(r_), bp(r_);
    nda::vector<complex *> cp(r_);
    for (int t = 0; t < r_; ++t) {
      ap(t) = left ? in.data() + std::size_t(t) * stride_ : op.data();
      bp(t) = left ? op.data() : in.data() + std::size_t(t) * stride_;
      cp(t) = out.data() + std::size_t(t) * stride_;
    }
    table->a = ap;
    table->b = bp;
    table->c = cp;
    tables_.emplace(key, std::move(table));
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
    nda::device_error_check(cudaMemsetAsync(out.data(), 0, checked_product(count_, sizeof(complex)), stream_), "dense CUDA zero");
  }
  void DeviceOps::copy(tensor_const_view in, tensor_view out) const {
    validate(in);
    validate(out);
    if (in.data() == out.data()) return;
    disjoint(in.data(), count_, out.data(), count_);
    nda::device_error_check(cudaMemcpyAsync(out.data(), in.data(), checked_product(count_, sizeof(complex)), cudaMemcpyDeviceToDevice, stream_),
                            "dense CUDA copy");
  }
  void DeviceOps::scale(complex alpha, tensor_view out) const {
    validate(out);
    auto scalar = nda::cucplx(alpha);
    check(cublasZscal(handle_, count_, &scalar, nda::cucplx(out.data()), 1), "scale");
  }
  void DeviceOps::axpy(complex alpha, tensor_const_view in, tensor_view out) const {
    validate(in);
    validate(out);
    disjoint(in.data(), count_, out.data(), count_);
    auto scalar = nda::cucplx(alpha);
    check(cublasZaxpy(handle_, count_, &scalar, nda::cucplx(in.data()), 1, nda::cucplx(out.data()), 1), "axpy");
  }
  void DeviceOps::batch_product(tensor_const_view a, tensor_const_view b, tensor_view out, complex alpha, complex beta) const {
    validate(a);
    validate(b);
    validate(out);
    disjoint(a.data(), count_, out.data(), count_);
    disjoint(b.data(), count_, out.data(), count_);
    // Row-major C=A*B becomes column-major C^T=B^T*A^T; T is not conjugation.
    auto av = nda::cucplx(alpha), bv = nda::cucplx(beta);
    if (mode_ == 2 && (N_ == 2 || N_ == 4 || N_ == 8 || N_ == 16 || N_ == 32 || N_ == 64)) {
      nda::device_error_check(xca_small_product(nda::cucplx(a.data()), nda::cucplx(b.data()), nda::cucplx(out.data()), r_, N_, av, bv, stream_),
                              "power-two product kernel");
      return;
    }
    check(cublasZgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N_, N_, N_, &av, nda::cucplx(b.data()), N_, stride_, nda::cucplx(a.data()), N_,
                                    stride_, &bv, nda::cucplx(out.data()), N_, stride_, r_),
          "batched product");
  }
  void DeviceOps::rank_product(matrix_const_view op, tensor_const_view in, tensor_view out, complex alpha, complex beta) const {
    validate(in);
    validate(out);
    require(op.shape() == std::array<long, 2>{rank_, rank_} && op.strides() == std::array<long, 2>{rank_, 1}, "invalid rank transform");
    disjoint(in.data(), count_, out.data(), count_);
    disjoint(op.data(), checked_product(rank_, rank_), out.data(), count_);
    // C-order Y=op*X becomes Y^T=X^T*op^T in column-major storage.
    auto av = nda::cucplx(alpha), bv = nda::cucplx(beta);
    if (rank_lanes_) {
      // Keep each lane's logical GEMM shape equal to the serial evaluator while
      // retaining interleaved storage. Leading dimensions still span all lanes.
      for (int lane = 0; lane < B_; ++lane)
        check(cublasZgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, stride_, rank_, rank_, &av,
                         nda::cucplx(in.data() + lane * stride_), stride_ * B_, nda::cucplx(op.data()), rank_, &bv,
                         nda::cucplx(out.data() + lane * stride_), stride_ * B_), "lane rank transform");
      return;
    }
    check(cublasZgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, stride_ * B_, rank_, rank_, &av, nda::cucplx(in.data()), stride_ * B_,
                      nda::cucplx(op.data()), rank_, &bv, nda::cucplx(out.data()), stride_ * B_),
          "rank transform");
  }
  void DeviceOps::vertex(matrix_const_view op, tensor_const_view in, tensor_view out, bool left, complex alpha, complex beta) {
    validate_operator(op);
    validate(in);
    validate(out);
    disjoint(op.data(), stride_, out.data(), count_);
    disjoint(in.data(), count_, out.data(), count_);
    auto key = Key{op.data(), in.data(), out.data(), left};
    if (!tables_.contains(key)) {
      prepare_vertex(op, in, out, left);
      wait_for_default_uploads();
    }
    auto const &tables = *tables_.at(key);
    auto av = nda::cucplx(alpha), bv = nda::cucplx(beta);
    if (mode_ == 2) {
      nda::device_error_check(xca_small_vertex(reinterpret_cast<cuDoubleComplex const *const *>(tables.b.data()),
        reinterpret_cast<cuDoubleComplex const *const *>(tables.a.data()),
        reinterpret_cast<cuDoubleComplex *const *>(tables.c.data()), r_, N_, av, bv, stream_), "small vertex kernel");
      return;
    }
    check(cublasZgemmBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N_, N_, N_, &av, reinterpret_cast<cuDoubleComplex const *const *>(tables.a.data()),
                             N_, reinterpret_cast<cuDoubleComplex const *const *>(tables.b.data()), N_, &bv,
                             reinterpret_cast<cuDoubleComplex *const *>(tables.c.data()), N_, r_),
          "vertex product");
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
  }
  void DeviceOps::traces(tensor_const_view in, matrix_const_view packed, matrix_view out) const {
    validate(in);
    int k = checked_blas_int(packed.extent(0));
    require(packed.extent(1) == stride_ && packed.strides() == std::array<long, 2>{stride_, 1}, "invalid packed trace operators");
    require(out.shape() == std::array<long, 2>{r_, k} && out.strides() == std::array<long, 2>{k, 1}, "invalid trace output");
    disjoint(in.data(), count_, out.data(), checked_product(r_, k));
    disjoint(packed.data(), checked_product(k, stride_), out.data(), checked_product(r_, k));
    // Y=A*packed^T: reinterpret C-order arrays in F order and transpose without
    // conjugation.
    auto one = nda::cucplx(complex{1.}), zero_value = nda::cucplx(complex{0.});
    check(cublasZgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, k, r_, stride_, &one, nda::cucplx(packed.data()), stride_, nda::cucplx(in.data()), stride_,
                      &zero_value, nda::cucplx(out.data()), k),
          "trace");
  }
  void DeviceOps::prepare_lanes(LaneTables &table, std::vector<complex const *> const &operators, tensor_const_view in, tensor_view out, bool left) {
    validate(in);
    validate(out);
    require(operators.size() == B_, "lane operator count mismatch");
    nda::vector<complex const *> ap(r_), bp(r_);
    nda::vector<complex *> cp(r_);
    for (int q = 0; q < r_; ++q) {
      ap(q) = left ? in.data() + std::size_t(q) * stride_ : operators[q % B_];
      bp(q) = left ? operators[q % B_] : in.data() + std::size_t(q) * stride_;
      cp(q) = out.data() + std::size_t(q) * stride_;
    }
    table.a = ap;
    table.b = bp;
    table.c = cp;
  }
  void DeviceOps::lane_vertex(LaneTables const &table, complex alpha, complex beta) const {
    auto av = nda::cucplx(alpha), bv = nda::cucplx(beta);
    if (mode_ == 2 && (N_ == 2 || N_ == 4 || N_ == 8 || N_ == 16 || N_ == 32 || N_ == 64)) {
      nda::device_error_check(xca_small_vertex(reinterpret_cast<cuDoubleComplex const *const *>(table.b.data()),
                                               reinterpret_cast<cuDoubleComplex const *const *>(table.a.data()),
                                               reinterpret_cast<cuDoubleComplex *const *>(table.c.data()), r_, N_, av, bv, stream_),
                              "power-two vertex kernel");
      return;
    }
    check(cublasZgemmBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N_, N_, N_, &av, reinterpret_cast<cuDoubleComplex const *const *>(table.a.data()), N_,
                             reinterpret_cast<cuDoubleComplex const *const *>(table.b.data()), N_, &bv,
                             reinterpret_cast<cuDoubleComplex *const *>(table.c.data()), N_, r_),
          "lane vertex");
  }
  void DeviceOps::broadcast(tensor_const_view input, tensor_view out, int active) const {
    require(input.shape() == std::array<long, 3>{rank_, N_, N_} && input.strides() == std::array<long, 3>{stride_, N_, 1}, "broadcast shape");
    require(active >= 0 && active <= B_, "broadcast lane count");
    validate(out);
    zero(out);
    for (int b = 0; b < active; ++b)
      nda::device_error_check(cudaMemcpy2DAsync(out.data() + std::size_t(b) * stride_, std::size_t(B_) * stride_ * sizeof(complex), input.data(),
                                                stride_ * sizeof(complex), stride_ * sizeof(complex), rank_, cudaMemcpyDeviceToDevice, stream_),
                              "broadcast lane");
  }
  void DeviceOps::zero_output(tensor_view out) const {
    require(out.shape() == std::array<long, 3>{rank_, N_, N_} && out.strides() == std::array<long, 3>{stride_, N_, 1}, "output shape");
    nda::device_error_check(cudaMemsetAsync(out.data(), 0, std::size_t(rank_) * stride_ * sizeof(complex), stream_), "zero output");
  }
  void DeviceOps::accumulate_lane(tensor_const_view input, tensor_view scratch, tensor_view sum, int lane, complex alpha) const {
    validate(input);
    require(lane >= 0 && lane < B_, "accumulation lane");
    require(scratch.shape() == std::array<long, 3>{rank_, N_, N_} && sum.shape() == scratch.shape(), "accumulation output shape");
    nda::device_error_check(cudaMemcpy2DAsync(scratch.data(), stride_ * sizeof(complex), input.data() + std::size_t(lane) * stride_,
                                              std::size_t(B_) * stride_ * sizeof(complex), stride_ * sizeof(complex), rank_, cudaMemcpyDeviceToDevice,
                                              stream_),
                            "gather lane");
    auto av = nda::cucplx(alpha);
    check(cublasZaxpy(handle_, rank_ * stride_, &av, nda::cucplx(scratch.data()), 1, nda::cucplx(sum.data()), 1), "ordered lane accumulation");
  }
  void DeviceOps::weighted_accumulate(tensor_const_view in, nda::cuvector_const_view<complex> w, nda::cuvector_const_view<complex> parity,
                                      tensor_view out) const {
    validate(in);
    validate(out);
    require(w.size() == r_ && parity.size() == r_ && w.strides()[0] == 1 && parity.strides()[0] == 1, "fused weights shape/contiguity");
    disjoint(in.data(), count_, out.data(), count_);
    nda::device_error_check(xca_small_weighted_add(nda::cucplx(in.data()), nda::cucplx(w.data()), nda::cucplx(parity.data()), nda::cucplx(out.data()),
                                                   count_, stride_, stream_),
                            "weighted add kernel");
  }
  void DeviceOps::axpy_scale(tensor_const_view in, tensor_view out, double scale) const {
    validate(in);
    validate(out);
    disjoint(in.data(), count_, out.data(), count_);
    nda::device_error_check(xca_small_add_scale(nda::cucplx(in.data()), nda::cucplx(out.data()), count_, scale, stream_), "add scale kernel");
  }
  void DeviceOps::axpy_flat(int count, complex alpha, complex const *in, complex *out) const {
    auto av = nda::cucplx(alpha);
    check(cublasZaxpy(handle_, count, &av, nda::cucplx(in), 1, nda::cucplx(out), 1), "flat accumulation");
  }
  void DeviceOps::clear_tables() {
    finish();
    tables_.clear();
  }
} // namespace triqs_xca::dense::cuda_small
#endif

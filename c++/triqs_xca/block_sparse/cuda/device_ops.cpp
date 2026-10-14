// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "device_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include "memory.hpp"
#ifdef XCA_HAVE_BLOCK_SPARSE_KERNELS
#include "kernels/kernels.hpp"
#endif
#include <cublas_v2.h>
#include <cusparse.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>

namespace triqs_xca::block_sparse::cuda {
  namespace {
    using Complex = nda::dcomplex;
    void check_cuda(cudaError_t s) {
      if (s != cudaSuccess) throw std::runtime_error(std::string("block-sparse CUDA: ") + cudaGetErrorString(s));
    }
    void check_blas(cublasStatus_t s) {
      if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("block-sparse cuBLAS status " + std::to_string(s));
    }
    void check_sparse(cusparseStatus_t s) {
      if (s != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("block-sparse cuSPARSE status " + std::to_string(s));
    }
    void require(bool condition, char const *message) {
      if (!condition) throw std::invalid_argument(std::string("block-sparse CUDA: ") + message);
    }
    int blas_int(long value) {
      require(value > 0 && value <= std::numeric_limits<int>::max(), "extent exceeds positive BLAS integer range");
      return static_cast<int>(value);
    }
    long extent(std::size_t n) {
      if (n > std::size_t(std::numeric_limits<long>::max())) throw std::overflow_error("block-sparse CUDA extent overflow");
      return static_cast<long>(n);
    }
    // Declare before device members. Cleanup selects their owning device; its
    // destructor restores the caller's device only after those members are gone.
    struct Context {
      int device = -1, restore = -1;
      Context() { check_cuda(cudaGetDevice(&device)); }
      void require_current() const {
        int current;
        check_cuda(cudaGetDevice(&current));
        require(current == device, "operation requires the construction device");
      }
      void begin_cleanup() noexcept {
        int current;
        if (cudaGetDevice(&current) == cudaSuccess && current != device && cudaSetDevice(device) == cudaSuccess) restore = current;
      }
      ~Context() {
        if (restore >= 0) cudaSetDevice(restore);
      }
    };
    std::size_t bytes(TensorConstView a) {
      auto s = a.shape();
      for (long n : s) require(n > 0, "tensor extents must be positive");
      auto plane = checked_multiply(s[1], s[2]);
      require(a.strides() == std::array<long, 3>{extent(plane), s[2], 1}, "tensor must be contiguous C-order");
      return tensor_bytes(s[0], s[1], s[2]);
    }
    void disjoint(TensorConstView a, TensorConstView b) {
      auto na = bytes(a), nb = bytes(b);
      auto x = reinterpret_cast<std::uintptr_t>(a.data()), y = reinterpret_cast<std::uintptr_t>(b.data());
      require(na <= UINTPTR_MAX - x && nb <= UINTPTR_MAX - y, "address range overflow");
      require(x + na <= y || y + nb <= x, "input and output overlap");
    }
    struct DenseDescriptor {
      cusparseDnMatDescr_t value = nullptr;
      ~DenseDescriptor() {
        if (value) cusparseDestroyDnMat(value);
      }
    };
    SparseBlock plain_transpose(SparseBlock const &host) {
      auto out = host.adjoint();
      for (auto &z : out.values) z = std::conj(z);
      return out;
    }
    std::size_t csr_bytes(SparseBlock const &a) {
      return checked_add(checked_multiply(checked_add(a.row_offsets.size(), a.column_indices.size()), sizeof(long)),
                         checked_multiply(a.values.size(), sizeof(Complex)));
    }
  } // namespace

  struct DeviceOps::Impl {
    Context context;
    cudaStream_t stream     = nullptr;
    cublasHandle_t blas     = nullptr;
    cusparseHandle_t sparse = nullptr;
    Impl() {
      try {
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check_blas(cublasCreate(&blas));
        check_blas(cublasSetStream(blas, stream));
        check_blas(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
        check_sparse(cusparseCreate(&sparse));
        check_sparse(cusparseSetStream(sparse, stream));
        check_sparse(cusparseSetPointerMode(sparse, CUSPARSE_POINTER_MODE_HOST));
      } catch (...) {
        cleanup();
        throw;
      }
    }
    void cleanup() noexcept {
      context.begin_cleanup();
      if (stream) cudaStreamSynchronize(stream);
      if (sparse) cusparseDestroy(sparse);
      if (blas) cublasDestroy(blas);
      if (stream) cudaStreamDestroy(stream);
    }
    ~Impl() { cleanup(); }
  };
  DeviceOps::DeviceOps() : impl_(std::make_unique<Impl>()) {}
  DeviceOps::~DeviceOps() = default;
  void DeviceOps::require_current() const { impl_->context.require_current(); }
  int DeviceOps::device() const { return impl_->context.device; }
  cudaStream_t DeviceOps::stream() const { return impl_->stream; }
  void DeviceOps::finish() const {
    require_current();
    check_cuda(cudaStreamSynchronize(stream()));
  }
  void DeviceOps::wait_for_default_uploads() const {
    require_current();
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    try {
      check_cuda(cudaEventRecord(event, nullptr));
      check_cuda(cudaStreamWaitEvent(stream(), event));
    } catch (...) {
      cudaEventDestroy(event);
      throw;
    }
    check_cuda(cudaEventDestroy(event));
  }
  void DeviceOps::zero(TensorView out) const {
    require_current();
    auto n = bytes(out);
    wait_for_default_uploads();
    check_cuda(cudaMemsetAsync(out.data(), 0, n, stream()));
    finish();
  }
  void DeviceOps::copy(TensorConstView in, TensorView out) const {
    require_current();
    require(in.shape() == out.shape(), "copy shape mismatch");
    auto n = bytes(in);
    (void)bytes(out);
    if (in.data() == out.data()) return;
    disjoint(in, out);
    wait_for_default_uploads();
    check_cuda(cudaMemcpyAsync(out.data(), in.data(), n, cudaMemcpyDeviceToDevice, stream()));
    finish();
  }
  void DeviceOps::axpy(Complex alpha, TensorConstView in, TensorView out) const {
    require_current();
    require(in.shape() == out.shape(), "axpy shape mismatch");
    disjoint(in, out);
    auto n = blas_int(extent(bytes(in) / sizeof(Complex)));
    wait_for_default_uploads();
    check_blas(cublasZaxpy(impl_->blas, n, reinterpret_cast<cuDoubleComplex const *>(&alpha), reinterpret_cast<cuDoubleComplex const *>(in.data()), 1,
                           reinterpret_cast<cuDoubleComplex *>(out.data()), 1));
    finish();
  }
  void DeviceOps::scale(Complex alpha, TensorView out) const {
    require_current();
    auto n = blas_int(extent(bytes(out) / sizeof(Complex)));
    wait_for_default_uploads();
    check_blas(cublasZscal(impl_->blas, n, reinterpret_cast<cuDoubleComplex const *>(&alpha), reinterpret_cast<cuDoubleComplex *>(out.data()), 1));
    finish();
  }
  void DeviceOps::weight_rows(TensorConstView in, nda::cuvector_const_view<Complex> weights, TensorView out) const {
    require_current();
    require(in.shape() == out.shape(), "weight shape mismatch");
    auto n = bytes(in);
    (void)bytes(out);
    int r = blas_int(in.extent(0)), width = blas_int(extent(n / sizeof(Complex) / r));
    require(weights.size() == r && weights.strides() == std::array<long, 1>{1}, "invalid rank weights");
    if (in.data() != out.data()) disjoint(in, out);
    auto x = reinterpret_cast<std::uintptr_t>(weights.data()), y = reinterpret_cast<std::uintptr_t>(out.data());
    auto nw = checked_multiply(r, sizeof(Complex));
    require(nw <= UINTPTR_MAX - x && n <= UINTPTR_MAX - y && (x + nw <= y || y + n <= x), "rank weights and output overlap");
    wait_for_default_uploads();
    check_blas(cublasZdgmm(impl_->blas, CUBLAS_SIDE_RIGHT, width, r, reinterpret_cast<cuDoubleComplex const *>(in.data()), width,
                           reinterpret_cast<cuDoubleComplex const *>(weights.data()), 1, reinterpret_cast<cuDoubleComplex *>(out.data()), width));
    finish();
  }
  void DeviceOps::pack_rhs(TensorConstView in, nda::cumatrix_view<Complex, nda::F_layout> out) const {
    require_current();
    auto n = bytes(in);
    int r = blas_int(in.extent(0)), width = blas_int(extent(n / sizeof(Complex) / r));
    require(out.shape() == std::array<long, 2>{r, width} && out.strides() == std::array<long, 2>{1, r}, "invalid RHS packing layout");
    auto x = reinterpret_cast<std::uintptr_t>(in.data()), y = reinterpret_cast<std::uintptr_t>(out.data());
    require(n <= UINTPTR_MAX - x && n <= UINTPTR_MAX - y && (x + n <= y || y + n <= x), "RHS packing overlaps");
    Complex one{1}, zero{0};
    wait_for_default_uploads();
    check_blas(cublasZgeam(impl_->blas, CUBLAS_OP_T, CUBLAS_OP_T, r, width, reinterpret_cast<cuDoubleComplex const *>(&one),
                           reinterpret_cast<cuDoubleComplex const *>(in.data()), width, reinterpret_cast<cuDoubleComplex const *>(&zero),
                           reinterpret_cast<cuDoubleComplex const *>(in.data()), width, reinterpret_cast<cuDoubleComplex *>(out.data()), r));
    finish();
  }
  void DeviceOps::unpack_rhs(nda::cumatrix_const_view<Complex, nda::F_layout> in, TensorView out) const {
    require_current();
    auto n = bytes(out);
    int r = blas_int(out.extent(0)), width = blas_int(extent(n / sizeof(Complex) / r));
    require(in.shape() == std::array<long, 2>{r, width} && in.strides() == std::array<long, 2>{1, r}, "invalid RHS unpacking layout");
    auto x = reinterpret_cast<std::uintptr_t>(in.data()), y = reinterpret_cast<std::uintptr_t>(out.data());
    require(n <= UINTPTR_MAX - x && n <= UINTPTR_MAX - y && (x + n <= y || y + n <= x), "RHS unpacking overlaps");
    Complex one{1}, zero{0};
    wait_for_default_uploads();
    check_blas(cublasZgeam(impl_->blas, CUBLAS_OP_T, CUBLAS_OP_T, width, r, reinterpret_cast<cuDoubleComplex const *>(&one),
                           reinterpret_cast<cuDoubleComplex const *>(in.data()), r, reinterpret_cast<cuDoubleComplex const *>(&zero),
                           reinterpret_cast<cuDoubleComplex const *>(in.data()), r, reinterpret_cast<cuDoubleComplex *>(out.data()), width));
    finish();
  }
  bool DeviceOps::any_abs_greater(TensorConstView in, double threshold) const {
    require_current();
    auto count = bytes(in) / sizeof(Complex);
    require(std::isfinite(threshold) && threshold >= 0, "invalid magnitude threshold");
    std::array<Complex, 4096> host;
    wait_for_default_uploads();
    for (std::size_t offset = 0; offset < count; offset += host.size()) {
      auto n = std::min(host.size(), count - offset);
      check_cuda(cudaMemcpyAsync(host.data(), in.data() + offset, n * sizeof(Complex), cudaMemcpyDeviceToHost, stream()));
      finish();
      for (std::size_t i = 0; i < n; ++i)
        if (std::abs(host[i]) > threshold) return true;
    }
    return false;
  }
  void DeviceOps::traces(TensorConstView in, TensorView out) const {
    require_current();
    disjoint(in, out);
    require(in.extent(1) == in.extent(2) && out.shape() == std::array<long, 3>{in.extent(0), 1, 1}, "trace shape mismatch");
    int d = blas_int(in.extent(1)), step = blas_int(in.extent(1) + 1);
    nda::vector<Complex> host(d);
    host() = Complex{1};
    nda::cuvector<Complex> ones(host);
    wait_for_default_uploads();
    check_blas(cublasSetPointerMode(impl_->blas, CUBLAS_POINTER_MODE_DEVICE));
    try {
      for (long t = 0; t < in.extent(0); ++t)
        check_blas(cublasZdotu(impl_->blas, d, reinterpret_cast<cuDoubleComplex const *>(in.data() + t * in.strides()[0]), step,
                               reinterpret_cast<cuDoubleComplex const *>(ones.data()), 1, reinterpret_cast<cuDoubleComplex *>(out.data() + t)));
      finish();
      check_blas(cublasSetPointerMode(impl_->blas, CUBLAS_POINTER_MODE_HOST));
    } catch (...) {
      cudaStreamSynchronize(stream());
      cublasSetPointerMode(impl_->blas, CUBLAS_POINTER_MODE_HOST);
      throw;
    }
  }
  void DeviceOps::add_trace(TensorConstView in, TensorView out, long mu, long kap) const {
    require_current();
    disjoint(in, out);
    require(in.shape() == std::array<long, 3>{out.extent(0), 1, 1} && mu >= 0 && mu < out.extent(1) && kap >= 0 && kap < out.extent(2),
            "trace slot mismatch");
    int r = blas_int(in.extent(0)), stride = blas_int(out.strides()[0]);
    Complex one{1};
    wait_for_default_uploads();
    check_blas(cublasZaxpy(impl_->blas, r, reinterpret_cast<cuDoubleComplex const *>(&one), reinterpret_cast<cuDoubleComplex const *>(in.data()), 1,
                           reinterpret_cast<cuDoubleComplex *>(out.data() + mu * out.extent(2) + kap), stride));
    finish();
  }
  void DeviceOps::product(TensorConstView left, TensorConstView right, TensorView out) const {
    require_current();
    disjoint(left, out);
    disjoint(right, out);
    auto a = left.shape(), b = right.shape(), c = out.shape();
    require(a[0] == b[0] && c == std::array<long, 3>{a[0], a[1], b[2]} && a[2] == b[1], "product shape mismatch");
    int r = blas_int(a[0]), M = blas_int(a[1]), N = blas_int(b[2]), K = blas_int(a[2]);
    Complex one{1}, zero{0};
    wait_for_default_uploads();
    // Row-major C=A*B is column-major C^T=B^T*A^T, with no conjugation.
    check_blas(cublasZgemmStridedBatched(
       impl_->blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, reinterpret_cast<cuDoubleComplex const *>(&one),
       reinterpret_cast<cuDoubleComplex const *>(right.data()), N, right.strides()[0], reinterpret_cast<cuDoubleComplex const *>(left.data()), K,
       left.strides()[0], reinterpret_cast<cuDoubleComplex const *>(&zero), reinterpret_cast<cuDoubleComplex *>(out.data()), N, out.strides()[0], r));
    finish();
  }
  void DeviceOps::rank_product(nda::cumatrix_const_view<Complex> op, TensorConstView in, TensorView out) const {
    require_current();
    require(in.shape() == out.shape(), "rank product shape mismatch");
    disjoint(in, out);
    long r = in.extent(0);
    require(op.shape() == std::array<long, 2>{r, r} && op.strides() == std::array<long, 2>{r, 1}, "rank matrix must be contiguous square C-order");
    auto x = reinterpret_cast<std::uintptr_t>(op.data()), y = reinterpret_cast<std::uintptr_t>(out.data());
    auto n = tensor_bytes(1, r, r), m = bytes(out);
    require(n <= UINTPTR_MAX - x && m <= UINTPTR_MAX - y && (x + n <= y || y + m <= x), "rank matrix and output overlap");
    int width = blas_int(extent(checked_multiply(in.extent(1), in.extent(2)))), rank = blas_int(r);
    Complex one{1}, zero{0};
    wait_for_default_uploads();
    check_blas(cublasZgemm(impl_->blas, CUBLAS_OP_N, CUBLAS_OP_N, width, rank, rank, reinterpret_cast<cuDoubleComplex const *>(&one),
                           reinterpret_cast<cuDoubleComplex const *>(in.data()), width, reinterpret_cast<cuDoubleComplex const *>(op.data()), rank,
                           reinterpret_cast<cuDoubleComplex const *>(&zero), reinterpret_cast<cuDoubleComplex *>(out.data()), width));
    finish();
  }

  bool mapped_special_available() noexcept { return ordered_sparse_available(); }
  bool ordered_sparse_available() noexcept {
#ifdef XCA_HAVE_BLOCK_SPARSE_KERNELS
    return true;
#else
    return false;
#endif
  }
  struct DeviceSparseBlock::Impl {
    Context context;
    std::shared_ptr<DeviceOps> ops;
    long rows, columns;
    std::size_t budget, resident;
    SparseBackend backend;
    struct Csr {
      nda::cuvector<long> offsets, indices;
      nda::cuvector<Complex> values;
      cusparseSpMatDescr_t descriptor = nullptr;
      explicit Csr(SparseBlock const &a)
         : offsets(extent(a.row_offsets.size())), indices(extent(a.column_indices.size())), values(extent(a.values.size())) {
        static_assert(sizeof(long) == 8);
        try {
          check_cuda(cudaMemcpy(offsets.data(), a.row_offsets.data(), a.row_offsets.size() * sizeof(long), cudaMemcpyHostToDevice));
          if (!a.values.empty()) {
            check_cuda(cudaMemcpy(indices.data(), a.column_indices.data(), a.column_indices.size() * sizeof(long), cudaMemcpyHostToDevice));
            check_cuda(cudaMemcpy(values.data(), a.values.data(), a.values.size() * sizeof(Complex), cudaMemcpyHostToDevice));
          }
          check_sparse(cusparseCreateCsr(&descriptor, a.rows, a.columns, extent(a.values.size()), offsets.data(), indices.data(), values.data(),
                                         CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
        } catch (...) {
          if (descriptor) cusparseDestroySpMat(descriptor);
          throw;
        }
      }
      ~Csr() {
        if (descriptor) cusparseDestroySpMat(descriptor);
      }
    };
    std::unique_ptr<Csr> forward, transposed;
    nda::cuvector<unsigned char> workspace;
    Impl(std::shared_ptr<DeviceOps> owner, SparseBlock const &host, std::size_t limit, SparseBackend selected)
       : ops(std::move(owner)), rows(host.rows), columns(host.columns), budget(limit), resident(0), backend(selected) {
      require(bool(ops), "missing device owner");
      ops->require_current();
      require(backend == SparseBackend::Library || backend == SparseBackend::Ordered, "invalid sparse backend");
      if (backend == SparseBackend::Ordered && !ordered_sparse_available())
        throw std::runtime_error("Ordered block-sparse CUDA kernels were not built");
      host.validate();
      require(rows > 0 && columns > 0, "cannot upload an absent/empty-dimension operator block");
      auto transpose = plain_transpose(host);
      resident       = checked_add(csr_bytes(host), csr_bytes(transpose));
      if (resident > budget) throw std::length_error("Sparse device block exceeds storage budget");
      forward    = std::make_unique<Csr>(host);
      transposed = std::make_unique<Csr>(transpose);
    }
    ~Impl() {
      context.begin_cleanup();
      try {
        ops->finish();
      } catch (...) {}
    }
    void apply(TensorConstView in, TensorView out, bool left) {
      ops->require_current();
      disjoint(in, out);
      auto a = in.shape(), b = out.shape();
      if (left)
        require(a[1] == columns && b == std::array<long, 3>{a[0], rows, a[2]}, "sparse left shape mismatch");
      else
        require(a[2] == rows && b == std::array<long, 3>{a[0], a[1], columns}, "sparse right shape mismatch");
      auto &csr = left ? *forward : *transposed;
      if (csr.values.size() == 0) {
        ops->zero(out);
        return;
      }
      long rhs = left ? a[2] : a[1], k = left ? columns : rows, m = left ? rows : columns;
#ifdef XCA_HAVE_BLOCK_SPARSE_KERNELS
      if (backend == SparseBackend::Ordered) {
        auto count = bytes(out) / sizeof(Complex);
        if ((count + 255) / 256 > std::size_t(std::numeric_limits<int>::max())) throw std::length_error("Sparse CUDA grid exceeds range");
        ops->wait_for_default_uploads();
        check_cuda(native::ordered_apply(csr.offsets.data(), csr.indices.data(), csr.values.data(), in.data(), out.data(), a[0], m, k, rhs, !left,
                                         ops->stream()));
        ops->finish(); // Caller tensors can be released immediately after return.
        return;
      }
#endif
      // Right product uses F^T * T^T: C-order T is a column-major T^T view.
      auto order = left ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
      long ld = left ? rhs : k, ldc = left ? rhs : m;
      Complex one{1}, zero{0};
      ops->wait_for_default_uploads();
      for (long t = 0; t < a[0]; ++t) {
        DenseDescriptor input, output;
        check_sparse(cusparseCreateDnMat(&input.value, k, rhs, ld, const_cast<Complex *>(in.data() + t * in.strides()[0]), CUDA_C_64F, order));
        check_sparse(cusparseCreateDnMat(&output.value, m, rhs, ldc, out.data() + t * out.strides()[0], CUDA_C_64F, order));
        std::size_t required = 0;
        check_sparse(cusparseSpMM_bufferSize(ops->impl_->sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                             csr.descriptor, input.value, &zero, output.value, CUDA_C_64F, CUSPARSE_SPMM_CSR_ALG3, &required));
        if (checked_add(resident, required) > budget) throw std::length_error("Sparse SpMM workspace exceeds storage budget");
        if (required > std::size_t(workspace.size())) {
          // All previous nodes complete before old workspace can be released.
          ops->finish();
          workspace = nda::cuvector<unsigned char>();
          workspace = nda::cuvector<unsigned char>(extent(required));
        }
        check_sparse(cusparseSpMM(ops->impl_->sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, csr.descriptor,
                                  input.value, &zero, output.value, CUDA_C_64F, CUSPARSE_SPMM_CSR_ALG3, workspace.data()));
        // The reference retains descriptors/workspace until their work completes.
        ops->finish();
      }
    }
  };
  DeviceSparseBlock::DeviceSparseBlock(std::shared_ptr<DeviceOps> ops, SparseBlock const &host, std::size_t budget, SparseBackend backend)
     : impl_(std::make_unique<Impl>(std::move(ops), host, budget, backend)) {}
  DeviceSparseBlock::~DeviceSparseBlock() = default;
  long DeviceSparseBlock::rows() const { return impl_->rows; }
  long DeviceSparseBlock::columns() const { return impl_->columns; }
  std::size_t DeviceSparseBlock::storage_bytes() const { return checked_add(impl_->resident, workspace_bytes()); }
  std::size_t DeviceSparseBlock::workspace_bytes() const { return impl_->workspace.size(); }
  void DeviceSparseBlock::apply_left(TensorConstView in, TensorView out) { impl_->apply(in, out, true); }
  void DeviceSparseBlock::apply_right(TensorConstView in, TensorView out) { impl_->apply(in, out, false); }
} // namespace triqs_xca::block_sparse::cuda
#endif

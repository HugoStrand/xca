// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "dlr_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include "memory.hpp"
#include <cusolverDn.h>
#include <cppdlr/dlr_kernels.hpp>
#include <cmath>
#include <limits>

namespace triqs_xca::block_sparse::cuda {
  namespace {
    using Complex = nda::dcomplex;
    void check_cuda(cudaError_t s) {
      if (s != cudaSuccess) throw std::runtime_error(std::string("block-sparse DLR CUDA: ") + cudaGetErrorString(s));
    }
    void check_solver(cusolverStatus_t s) {
      if (s != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("block-sparse cuSOLVER status " + std::to_string(s));
    }
    long extent(std::size_t n) {
      if (n > std::size_t(std::numeric_limits<int>::max())) throw std::overflow_error("DLR extent exceeds solver/BLAS range");
      return n;
    }
    struct Context {
      int device = -1, restore = -1;
      Context() { check_cuda(cudaGetDevice(&device)); }
      void begin_cleanup() noexcept {
        int current;
        if (cudaGetDevice(&current) == cudaSuccess && current != device && cudaSetDevice(device) == cudaSuccess) restore = current;
      }
      ~Context() {
        if (restore >= 0) cudaSetDevice(restore);
      }
    };
  } // namespace
  struct DlrOps::Impl {
    Context context; // Outlives all device members and restores device last.
    std::shared_ptr<DeviceOps> ops;
    long r;
    std::size_t budget, tables;
    cusolverDnHandle_t solver = nullptr;
    nda::cumatrix<Complex> lu, C, H, Q, reflection;
    nda::cuvector<int> pivots, status;
    nda::cumatrix<Complex, nda::F_layout> rhs;
    Tensor hf, hg, product, cross, diagonal;
    std::array<long, 3> convolution_shape{};
    Impl(std::shared_ptr<DeviceOps> owner, cppdlr::imtime_ops const &host, std::size_t limit)
       : ops(std::move(owner)), r(host.rank()), budget(limit), tables(0) {
      if (!ops) throw std::invalid_argument("Missing DLR device owner");
      ops->require_current();
      if (r <= 0) throw std::invalid_argument("Empty DLR rank");
      (void)extent(r);
      tables = checked_add(checked_multiply(5, tensor_bytes(1, r, r)), checked_multiply(checked_add(r, 1), sizeof(int)));
      admit(tables);
      nda::matrix<Complex> c(host.get_cf2it()), h(r, r), q(r, r);
      auto rf = host.get_rfnodes(), it = host.get_itnodes();
      for (long j = 0; j < r; ++j)
        for (long k = 0; k < r; ++k) {
          h(j, k) = j == k ? 0. : cppdlr::k_it(0., rf(k)) / (rf(k) - rf(j));
          q(j, k) = (it(j) > 0 ? it(j) : 1. + it(j)) * c(j, k) * cppdlr::k_it(0., rf(k));
        }
      auto identity = nda::matrix<Complex>::zeros({r, r});
      for (long i = 0; i < r; ++i) identity(i, i) = 1.;
      lu         = nda::cumatrix<Complex>(host.get_it2cf_zlu());
      pivots     = nda::cuvector<int>(host.get_it2cf_piv());
      status     = nda::cuvector<int>(1);
      C          = nda::cumatrix<Complex>(c);
      H          = nda::cumatrix<Complex>(h);
      Q          = nda::cumatrix<Complex>(q);
      reflection = nda::cumatrix<Complex>(host.reflect(identity));
      ops->wait_for_default_uploads();
      ops->finish();
      try {
        check_solver(cusolverDnCreate(&solver));
        check_solver(cusolverDnSetStream(solver, ops->stream()));
      } catch (...) {
        if (solver) cusolverDnDestroy(solver);
        throw;
      }
    }
    ~Impl() {
      context.begin_cleanup();
      try {
        ops->finish();
      } catch (...) {}
      if (solver) cusolverDnDestroy(solver);
    }
    void admit(std::size_t n) const {
      if (n > budget) throw std::length_error("Block-sparse DLR explicit storage exceeds budget");
    }
    std::size_t scratch_bytes() const {
      std::size_t n = 0;
      for (auto const *a : {&hf, &hg, &product, &cross, &diagonal}) n = checked_add(n, checked_multiply(a->size(), sizeof(Complex)));
      return n;
    }
    std::size_t storage_bytes() const { return checked_add(tables, checked_add(scratch_bytes(), checked_multiply(rhs.size(), sizeof(Complex)))); }
    void validate(TensorConstView a) const {
      ops->require_current();
      auto s = a.shape();
      if (s[0] != r || s[1] <= 0 || s[2] <= 0) throw std::invalid_argument("Invalid DLR tensor shape");
      auto width = extent(checked_multiply(s[1], s[2]));
      // The reference uses one BLAS vector for its final axpy/scal. Reject the
      // corresponding limit before solving or writing any caller output.
      (void)extent(checked_multiply(r, width));
      if (a.strides() != std::array<long, 3>{width, s[2], 1}) throw std::invalid_argument("DLR tensor must be contiguous C-order");
    }
    void rhs_for(long width) {
      if (rhs.shape() == std::array<long, 2>{r, width}) return;
      admit(checked_add(tables, checked_add(scratch_bytes(), tensor_bytes(1, r, width))));
      ops->finish();
      rhs = nda::cumatrix<Complex, nda::F_layout>();
      rhs = nda::cumatrix<Complex, nda::F_layout>(r, width);
    }
    void scratch_for(long M, long K, long N) {
      auto shape = std::array<long, 3>{M, K, N};
      if (shape == convolution_shape) return;
      auto amount = checked_add(tensor_bytes(r, M, K), tensor_bytes(r, K, N));
      amount      = checked_add(amount, checked_multiply(3, tensor_bytes(r, M, N)));
      admit(checked_add(tables, checked_add(amount, checked_multiply(rhs.size(), sizeof(Complex)))));
      ops->finish();
      convolution_shape = {};
      hf                = Tensor();
      hg                = Tensor();
      product           = Tensor();
      cross             = Tensor();
      diagonal          = Tensor();
      hf                = Tensor(r, M, K);
      hg                = Tensor(r, K, N);
      product           = Tensor(r, M, N);
      cross             = Tensor(r, M, N);
      diagonal          = Tensor(r, M, N);
      convolution_shape = shape;
    }
  };
  DlrOps::DlrOps(std::shared_ptr<DeviceOps> ops, cppdlr::imtime_ops const &host, std::size_t budget)
     : impl_(std::make_unique<Impl>(std::move(ops), host, budget)) {}
  DlrOps::~DlrOps() = default;
  void DlrOps::release_scratch() {
    auto &x = *impl_;
    x.ops->require_current();
    if (x.rhs.size() == 0 && x.scratch_bytes() == 0) return;
    x.ops->finish();
    x.rhs               = decltype(x.rhs)();
    x.hf                = Tensor();
    x.hg                = Tensor();
    x.product           = Tensor();
    x.cross             = Tensor();
    x.diagonal          = Tensor();
    x.convolution_shape = {};
  }
  long DlrOps::rank() const { return impl_->r; }
  std::size_t DlrOps::storage_bytes() const { return impl_->storage_bytes(); }
  void DlrOps::vals2coefs(TensorConstView values, TensorView coefficients) {
    auto &x = *impl_;
    x.validate(values);
    x.validate(coefficients);
    if (values.shape() != coefficients.shape()) throw std::invalid_argument("DLR solve shape mismatch");
    auto width = extent(checked_multiply(values.extent(1), values.extent(2)));
    x.rhs_for(width);
    x.ops->pack_rhs(values, x.rhs);
    // cppdlr/NDA factorized the C-layout cached matrix as its column-major
    // transpose; solve with T (not C), preserving its pivot/factor convention.
    check_solver(cusolverDnZgetrs(x.solver, CUBLAS_OP_T, x.r, width, reinterpret_cast<cuDoubleComplex const *>(x.lu.data()), x.r, x.pivots.data(),
                                  reinterpret_cast<cuDoubleComplex *>(x.rhs.data()), x.r, x.status.data()));
    int info = -1;
    check_cuda(cudaMemcpyAsync(&info, x.status.data(), sizeof(int), cudaMemcpyDeviceToHost, x.ops->stream()));
    x.ops->finish();
    if (info != 0) throw std::runtime_error("Block-sparse DLR getrs info " + std::to_string(info));
    x.ops->unpack_rhs(x.rhs, coefficients);
  }
  void DlrOps::coefs2vals(TensorConstView coefficients, TensorView values) const {
    impl_->validate(coefficients);
    impl_->validate(values);
    impl_->ops->rank_product(impl_->C, coefficients, values);
  }
  void DlrOps::reflect(TensorConstView values, TensorView reflected) const {
    impl_->validate(values);
    impl_->validate(reflected);
    impl_->ops->rank_product(impl_->reflection, values, reflected);
  }
  void DlrOps::convolve(double beta, TensorConstView fc, TensorConstView gc, TensorView out) {
    auto &x = *impl_;
    x.validate(fc);
    x.validate(gc);
    x.validate(out);
    long M = fc.extent(1), K = fc.extent(2), N = gc.extent(2);
    if (!std::isfinite(beta) || gc.extent(1) != K || out.shape() != std::array<long, 3>{x.r, M, N})
      throw std::invalid_argument("Invalid rectangular DLR convolution inputs");
    x.scratch_for(M, K, N);
    x.ops->rank_product(x.H, fc, x.hf);
    x.ops->rank_product(x.H, gc, x.hg);
    x.ops->product(fc, gc, x.product);
    x.ops->rank_product(x.Q, x.product, x.diagonal);
    x.ops->product(x.hf, gc, x.cross);
    x.ops->product(fc, x.hg, x.product);
    x.ops->axpy(1., x.product, x.cross);
    x.ops->rank_product(x.C, x.cross, x.product);
    x.ops->copy(x.diagonal, out);
    x.ops->axpy(1., x.product, out);
    x.ops->scale(beta, out);
  }
} // namespace triqs_xca::block_sparse::cuda
#endif

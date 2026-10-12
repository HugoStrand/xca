// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "dlr_ops.hpp"
#include "profile_range.hpp"
#ifdef NDA_HAVE_CUDA
#include <cmath>
#include <cstdlib>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/lapack.hpp>
#include <stdexcept>
#include <string>

namespace xca_experiments::power_two {

  DlrOps::DlrOps(cppdlr::imtime_ops const &host, long dimension, int capacity)
     : r_(DeviceOps::checked_blas_int(host.rank())),
       N_(DeviceOps::checked_blas_int(dimension)),
       columns_(DeviceOps::checked_blas_int(DeviceOps::checked_product(capacity, DeviceOps::checked_product(N_, N_)))),
       B_(capacity),
       ops_(r_, N_, B_),
       lu_(host.get_it2cf_zlu()),
       pivots_(host.get_it2cf_piv()),
       rhs_(r_, columns_),
       hf_(r_ * B_, N_, N_),
       hg_(r_ * B_, N_, N_),
       product_(r_ * B_, N_, N_),
       cross_(r_ * B_, N_, N_),
       status_(DeviceOps::checked_product(256, capacity)) {
    if (auto value = std::getenv("XCA_POW2_SOLVE_LANES")) {
      if (std::string(value) != "0" && std::string(value) != "1") throw std::invalid_argument("solve lanes must be 0 or 1");
      solve_group_ = std::string(value) == "1" ? 1 : 0;
    }
    if (auto value = std::getenv("XCA_POW2_SOLVE_GROUP")) {
      std::string text(value);
      std::size_t consumed = 0;
      int group = std::stoi(text, &consumed);
      if (consumed != text.size() || group < 1 || group > 128) throw std::invalid_argument("solve group must be 1..128");
      solve_group_ = group;
    }
    // Preserve the cached LU's C layout and NDA's corresponding getrs transpose
    // convention. Reordering factors as if they were the unfactored matrix is
    // wrong.
    nda::matrix<complex> C(host.get_cf2it()), H(r_, r_), Q(r_, r_);
    auto rf = host.get_rfnodes(), it = host.get_itnodes();
    for (int j = 0; j < r_; ++j)
      for (int k = 0; k < r_; ++k) {
        H(j, k) = j == k ? 0. : cppdlr::k_it(0., rf(k)) / (rf(k) - rf(j));
        Q(j, k) = (it(j) > 0 ? it(j) : 1. + it(j)) * C(j, k) * cppdlr::k_it(0., rf(k));
      }
    auto identity = nda::matrix<complex>::zeros({r_, r_});
    for (int i = 0; i < r_; ++i) identity(i, i) = 1.;
    C_          = nda::cumatrix<complex>(C);
    H_          = nda::cumatrix<complex>(H);
    Q_          = nda::cumatrix<complex>(Q);
    reflection_ = nda::cumatrix<complex>(host.reflect(identity));
    ops_.wait_for_default_uploads();
    if (cusolverDnCreate(&solver_) != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("solver create failed");
    if (cusolverDnSetStream(solver_, ops_.stream()) != CUSOLVER_STATUS_SUCCESS) {
      cusolverDnDestroy(solver_);
      solver_ = nullptr;
      throw std::runtime_error("solver stream failed");
    }
  }
  DlrOps::~DlrOps() {
    try {
      finish();
    } catch (...) {}
    if (solver_) cusolverDnDestroy(solver_);
  }
  void DlrOps::finish() {
    ops_.finish();
    for (int i = 0; i < used_status_; ++i)
      if (status_(i) != 0) throw std::runtime_error("serial_stream getrs info " + std::to_string(status_(i)));
    used_status_ = 0;
  }

  std::size_t DlrOps::workspace_bytes() const {
    auto mul      = DeviceOps::checked_product;
    auto add      = DeviceOps::checked_sum;
    auto elements = add(mul(5, mul(r_, r_)), mul(5, mul(r_, columns_)));
    return add(status_.size() * sizeof(int), add(ops_.workspace_bytes(), add(mul(elements, sizeof(complex)), mul(r_, sizeof(int)))));
  }
  void DlrOps::validate(tensor_const_view input) const {
    if (input.shape() != std::array<long, 3>{r_ * B_, N_, N_} || input.strides() != std::array<long, 3>{N_ * N_, N_, 1})
      throw std::invalid_argument("dense CUDA DLR: expected contiguous (r,N,N) tensor");
  }

  void DlrOps::vals2coefs(tensor_const_view values, tensor_view coefficients) {
    ProfileRange phase("xca.phase.vals2coefs");
    validate(values);
    validate(coefficients);
    int group = solve_group_ ? std::min(solve_group_, B_) : B_;
    int calls = (B_ + group - 1) / group;
    if (calls > status_.size() - used_status_) throw std::invalid_argument("solver status capacity exceeded");
    ops_.to_fortran(matrix_const_view{nda::reshape(values, r_, columns_)}, rhs_);
    for (int first = 0; first < B_; first += group) {
      int width = std::min(group, B_ - first) * N_ * N_;
      auto result = cusolverDnZgetrs(solver_, CUBLAS_OP_T, r_, width, nda::cucplx(lu_.data()), r_, pivots_.data(),
                                    nda::cucplx(rhs_.data() + long(first) * N_ * N_ * r_), r_, status_.data() + used_status_++);
      if (result != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("getrs submission failed");
    }
    ops_.from_fortran(rhs_, matrix_view{nda::reshape(coefficients, r_, columns_)});
  }

  void DlrOps::coefs2vals(tensor_const_view coefficients, tensor_view values) const { ops_.rank_product(C_, coefficients, values); }

  void DlrOps::reflect(tensor_const_view values, tensor_view reflected) const {
    ProfileRange phase("xca.phase.reflection");
    ops_.rank_product(reflection_, values, reflected);
  }

  void DlrOps::convolve(double beta, tensor_const_view fc, tensor_const_view gc, tensor_view out) {
    ProfileRange phase("xca.phase.convolution");
    validate(fc);
    validate(gc);
    validate(out);
    if (!std::isfinite(beta)) throw std::invalid_argument("dense CUDA DLR: beta must be finite");
    ops_.rank_product(H_, fc, hf_);
    ops_.rank_product(H_, gc, hg_);
    ops_.batch_product(fc, gc, product_);
    ops_.batch_product(hf_, gc, cross_);
    // Match cppdlr's addition and final beta scaling order. Reuse consumed
    // scratch instead of fusing independent products/terms through GEMM beta;
    // this limits differences amplified by repeated ill-conditioned DLR solves.
    ops_.batch_product(fc, hg_, hf_);
    ops_.axpy(1., hf_, cross_);
    ops_.rank_product(Q_, product_, out);
    ops_.rank_product(C_, cross_, product_);
    if (ops_.mode())
      ops_.axpy_scale(product_, out, beta);
    else {
      ops_.axpy(1., product_, out);
      ops_.scale(beta, out);
    }
  }

} // namespace xca_experiments::power_two
#endif

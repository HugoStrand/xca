// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "dlr_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include <nda/lapack.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace triqs_xca::dense::cuda {

  DlrOps::DlrOps(cppdlr::imtime_ops const &host, long dimension)
     : r_(DeviceOps::checked_blas_int(host.rank())),
       N_(DeviceOps::checked_blas_int(dimension)),
       columns_(DeviceOps::checked_blas_int(DeviceOps::checked_product(N_, N_))),
       ops_(r_, N_),
       lu_(host.get_it2cf_zlu()),
       pivots_(host.get_it2cf_piv()),
       rhs_(r_, columns_),
       hf_(r_, N_, N_),
       hg_(r_, N_, N_),
       product_(r_, N_, N_),
       cross_(r_, N_, N_) {
    // Preserve the cached LU's C layout and NDA's corresponding getrs transpose
    // convention. Reordering factors as if they were the unfactored matrix is wrong.
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
    // Initialize NDA's lazy shared BLAS/solver handles and one-word unified
    // status before diagram traversal. All buffers used here are already owned.
    // NDA's solver handle/status are process-shared; callers must serialize
    // device solver use, including use through other NDA users.
    if (!nda::lapack::device::get_synchronization()) throw std::runtime_error("dense CUDA DLR: NDA solver synchronization must be enabled");
    ops_.zero(hf_);
    ops_.scale(1., hf_);
    auto rhs_bytes = DeviceOps::checked_product(DeviceOps::checked_product(r_, columns_), sizeof(complex));
    nda::device_error_check(cudaMemset(rhs_.data(), 0, rhs_bytes), "dense CUDA DLR solver warmup");
    int info = nda::lapack::getrs(lu_, rhs_, pivots_);
    if (info != 0) throw std::runtime_error("dense CUDA DLR solver warmup: info " + std::to_string(info));
  }

  void DlrOps::validate(tensor_const_view input) const {
    if (input.shape() != std::array<long, 3>{r_, N_, N_} || input.strides() != std::array<long, 3>{columns_, N_, 1})
      throw std::invalid_argument("dense CUDA DLR: expected contiguous (r,N,N) tensor");
  }

  void DlrOps::vals2coefs(tensor_const_view values, tensor_view coefficients) {
    validate(values);
    validate(coefficients);
    // The NDA wrapper reads its unified-memory status before returning. Its
    // synchronization must be enabled; do not mutate the caller's shared setting.
    if (!nda::lapack::device::get_synchronization()) throw std::runtime_error("dense CUDA DLR: NDA solver synchronization must be enabled");
    ops_.to_fortran(matrix_const_view{nda::reshape(values, r_, columns_)}, rhs_);
    int info = nda::lapack::getrs(lu_, rhs_, pivots_);
    if (info != 0) throw std::runtime_error("dense CUDA DLR getrs: info " + std::to_string(info));
    ops_.from_fortran(rhs_, matrix_view{nda::reshape(coefficients, r_, columns_)});
  }

  void DlrOps::coefs2vals(tensor_const_view coefficients, tensor_view values) const { ops_.rank_product(C_, coefficients, values); }

  void DlrOps::reflect(tensor_const_view values, tensor_view reflected) const { ops_.rank_product(reflection_, values, reflected); }

  void DlrOps::convolve(double beta, tensor_const_view fc, tensor_const_view gc, tensor_view out) {
    validate(fc);
    validate(gc);
    validate(out);
    if (!std::isfinite(beta)) throw std::invalid_argument("dense CUDA DLR: beta must be finite");
    ops_.rank_product(H_, fc, hf_);
    ops_.rank_product(H_, gc, hg_);
    ops_.batch_product(fc, gc, product_);
    ops_.batch_product(hf_, gc, cross_);
    ops_.batch_product(fc, hg_, cross_, 1., 1.);
    ops_.rank_product(Q_, product_, out, beta, 0.);
    ops_.rank_product(C_, cross_, out, beta, 1.);
  }

} // namespace triqs_xca::dense::cuda
#endif

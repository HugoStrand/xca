// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernels.hpp"
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
namespace scalar_edge_experiment {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  struct ScalarPack {
    std::shared_ptr<gpu::DeviceOps> owner;
    gpu::DlrOps dlr;
    nda::cumatrix<Complex> C, H, Q;
    gpu::Tensor f, g, fc, gc, hf, hg, tmp, diagonal, out;
    ScalarPack(std::shared_ptr<gpu::DeviceOps> ops, cppdlr::imtime_ops const &host, long lanes) : owner(ops), dlr(ops, host) {
      long r = host.rank();
      nda::matrix<Complex> h(r, r), q(r, r);
      auto rf = host.get_rfnodes(), it = host.get_itnodes();
      for (long j = 0; j < r; ++j)
        for (long k = 0; k < r; ++k) {
          h(j, k) = j == k ? 0. : cppdlr::k_it(0., rf(k)) / (rf(k) - rf(j));
          q(j, k) = (it(j) > 0 ? it(j) : 1. + it(j)) * host.get_cf2it()(j, k) * cppdlr::k_it(0., rf(k));
        }
      C = nda::cumatrix<Complex>(nda::matrix<Complex>(host.get_cf2it()));
      H = nda::cumatrix<Complex>(h);
      Q = nda::cumatrix<Complex>(q);
      for (auto *a : {&f, &g, &fc, &gc, &hf, &hg, &tmp, &diagonal, &out}) *a = gpu::Tensor(r, lanes, 1);
      owner->wait_for_default_uploads();
      owner->finish();
    }
    void compute(double beta = 2.) {
      dlr.vals2coefs(f, fc);
      dlr.vals2coefs(g, gc);
      owner->rank_product(H, fc, hf);
      owner->rank_product(H, gc, hg);
      check(scalar_edge_experiment::multiply(fc.data(), gc.data(), tmp.data(), tmp.size(), owner->stream()));
      owner->rank_product(Q, tmp, diagonal);
      check(scalar_edge_experiment::cross(hf.data(), gc.data(), fc.data(), hg.data(), tmp.data(), tmp.size(), owner->stream()));
      owner->rank_product(C, tmp, out);
      owner->axpy(1., out, diagonal);
      owner->scale(beta, diagonal);
    }
    std::size_t bytes() const { return dlr.storage_bytes() + (C.size() + H.size() + Q.size() + 9 * f.size()) * sizeof(Complex); }
  };
} // namespace scalar_edge_experiment

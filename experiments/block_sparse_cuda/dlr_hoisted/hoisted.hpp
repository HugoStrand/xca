// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cppdlr/cppdlr.hpp>

namespace block_sparse_dlr_experiment {
  using Array = nda::array<nda::dcomplex, 3>;
  // Experimental time-ordered CPU control. Preserve cppdlr products and their
  // addition order, but compute each whole-input Hilbert transform only once.
  class Hoisted {
    nda::matrix<double> C, H, Q;

    public:
    explicit Hoisted(cppdlr::imtime_ops const &host) : C(host.get_cf2it()), H(host.rank(), host.rank()), Q(host.rank(), host.rank()) {
      auto rf = host.get_rfnodes(), it = host.get_itnodes();
      for (long j = 0; j < host.rank(); ++j)
        for (long k = 0; k < host.rank(); ++k) {
          H(j, k) = j == k ? 0. : cppdlr::k_it(0., rf(k)) / (rf(k) - rf(j));
          Q(j, k) = (it(j) > 0 ? it(j) : 1. + it(j)) * C(j, k) * cppdlr::k_it(0., rf(k));
        }
    }
    Array convolve(double beta, Array const &fc, Array const &gc) const {
      if (fc.shape(0) != C.shape(0) || gc.shape(0) != C.shape(0) || fc.shape(2) != gc.shape(1))
        throw std::invalid_argument("Invalid hoisted convolution shape");
      constexpr auto all = nda::range::all;
      auto r             = C.shape(0);
      Array tmp(r, fc.shape(1), gc.shape(2));
      for (long i = 0; i < r; ++i) tmp(i, all, all) = cppdlr::matmul(fc(i, all, all), gc(i, all, all));
      auto h  = cppdlr::arraymult(Q, tmp);
      auto hf = cppdlr::arraymult(H, fc), hg = cppdlr::arraymult(H, gc);
      for (long i = 0; i < r; ++i)
        tmp(i, all, all) = cppdlr::matmul(hf(i, all, all), gc(i, all, all)) + cppdlr::matmul(fc(i, all, all), hg(i, all, all));
      return beta * (h + cppdlr::arraymult(C, tmp));
    }
  };
  // Private generated-evaluator adapter: all inherited mesh/solve operations stay
  // unchanged. Only the two time-ordered rank-three convolution calls differ.
  class ImtimeControl : public cppdlr::imtime_ops {
    Hoisted control;

    public:
    explicit ImtimeControl(cppdlr::imtime_ops const &host) : cppdlr::imtime_ops(host), control(host) {}
    Array convolve(double beta, Array const &fc, Array const &gc, bool time_order) const {
      if (!time_order) throw std::invalid_argument("The control supports time-ordered convolution only");
      return control.convolve(beta, fc, gc);
    }
  };
} // namespace block_sparse_dlr_experiment

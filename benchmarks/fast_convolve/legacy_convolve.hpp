// Copyright (c) 2022-2023 Simons Foundation
// Copyright (c) 2023 Hugo Strand
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Original authors: Hugo U. R. Strand, Nils Wentzell, Jason Kaye
//
// Benchmark-only adaptation of cppdlr/c++/cppdlr/dlr_imtime.hpp at
// 7a0f60cba2da69d64361971613b6859bb8c078ce, before fix b9b24ab.
// Retains the rank-3 time-ordered convolution's repeated whole-array Hilbert
// transforms inside the DLR-node loop, including their original temporaries.
// The only specialization is to complex rank-3 inputs and time ordering;
// fixed helper matrices are initialized in the constructor, outside timings.

#pragma once

#include <cppdlr/dlr_imtime.hpp>

#include <stdexcept>

namespace benchmark {

  class legacy_convolve {
    cppdlr::imtime_ops const &itops;
    nda::matrix<double> thilb, ttcf2it;

    public:
    explicit legacy_convolve(cppdlr::imtime_ops const &itops) : itops(itops), thilb(itops.rank(), itops.rank()), ttcf2it(itops.rank(), itops.rank()) {
      auto const r = itops.rank();
      auto dlr_rf  = itops.get_rfnodes();
      auto dlr_it  = itops.get_itnodes();
      auto cf2it   = itops.get_cf2it();

      // Original tconvolve_init, with its real helper matrices unchanged.
      for (int j = 0; j < r; ++j) {
        for (int k = 0; k < r; ++k) {
          if (j == k) {
            thilb(j, k) = 0;
          } else {
            thilb(j, k) = cppdlr::k_it(0.0, dlr_rf(k)) / (dlr_rf(k) - dlr_rf(j));
          }
        }
      }
      for (int j = 0; j < r; ++j) {
        for (int k = 0; k < r; ++k) {
          if (dlr_it(j) > 0) {
            ttcf2it(j, k) = dlr_it(j) * cf2it(j, k) * cppdlr::k_it(0.0, dlr_rf(k));
          } else {
            ttcf2it(j, k) = (1 + dlr_it(j)) * cf2it(j, k) * cppdlr::k_it(0.0, dlr_rf(k));
          }
        }
      }
    }

    legacy_convolve(cppdlr::imtime_ops &&) = delete;

    nda::array<nda::dcomplex, 3> convolve(double beta, nda::array<nda::dcomplex, 3> const &fc, nda::array<nda::dcomplex, 3> const &gc) const {
      using cppdlr::arraymult;
      using cppdlr::matmul;
      constexpr auto _ = nda::range::all;
      int const r      = itops.rank();
      if (r != fc.shape(0) || r != gc.shape(0)) throw std::runtime_error("First dim of input arrays must be equal to DLR rank r.");
      if (fc.shape(2) != gc.shape(1)) throw std::runtime_error("Input array dimensions incompatible.");

      auto hilb_v   = nda::matrix_const_view<double>(thilb);
      auto tcf2it_v = nda::matrix_const_view<double>(ttcf2it);
      auto cf2it    = itops.get_cf2it();

      auto tmp = nda::array<nda::dcomplex, 3>(r, fc.shape(1), gc.shape(2));
      for (int i = 0; i < r; ++i) { tmp(i, _, _) = matmul(fc(i, _, _), gc(i, _, _)); }
      auto h = arraymult(tcf2it_v, tmp);

      // Intentionally reproduce the old O(r^3 D^2) redundant transform work.
      for (int i = 0; i < r; ++i) {
        tmp(i, _, _) = matmul(arraymult(hilb_v, fc)(i, _, _), gc(i, _, _)) + matmul(fc(i, _, _), arraymult(hilb_v, gc)(i, _, _));
      }

      return beta * (h + arraymult(cf2it, tmp));
    }
  };

} // namespace benchmark

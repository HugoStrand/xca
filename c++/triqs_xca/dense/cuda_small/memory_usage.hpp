// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "device_ops.hpp"
#ifdef XCA_HAVE_CUDA_SMALL
#include <cppdlr/dlr_imtime.hpp>
#include <triqs_xca/dense/fset.hpp>
#include <triqs_xca/hyb.hpp>

namespace triqs_xca::dense::cuda_small {
  template <typename Array> std::size_t array_bytes(Array const &a) {
    return DeviceOps::checked_product(a.size(), sizeof(typename Array::value_type));
  }
  inline std::size_t hybridization_host_bytes(hyb::Hybridization const &h) {
    auto add = DeviceOps::checked_sum;
    return add(add(add(array_bytes(h.poles), array_bytes(h.coeffs)), add(array_bytes(h.values), array_bytes(h.values_reflect))),
               add(add(array_bytes(h.k_0_p), array_bytes(h.k_0_m)), add(array_bytes(h.k_it_p), array_bytes(h.k_it_m))));
  }
  inline std::size_t operator_host_bytes(dense::FSet const &f) {
    auto add = DeviceOps::checked_sum;
    return add(add(array_bytes(f.Fs), array_bytes(f.F_dags)), add(array_bytes(f.F_dag_bars), array_bytes(f.F_bars_refl)));
  }
  inline std::size_t imtime_host_bound_bytes(cppdlr::imtime_ops const &h) {
    auto add = DeviceOps::checked_sum;
    auto known = add(add(array_bytes(h.get_rfnodes()), array_bytes(h.get_itnodes())),
                     add(add(array_bytes(h.get_cf2it()), array_bytes(h.get_it2cf_lu())),
                         add(array_bytes(h.get_it2cf_zlu()), array_bytes(h.get_it2cf_piv()))));
    // cppdlr's six private lazy real r-by-r caches may arrive populated in a
    // copied caller snapshot. Count their maximum without materializing them.
    auto caches = DeviceOps::checked_product(6 * sizeof(double), DeviceOps::checked_product(h.rank(), h.rank()));
    return add(known, caches);
  }
}
#endif

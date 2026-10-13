// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdexcept>

namespace triqs_xca {
  // A half-open range over the ordinary flat-index domain, including empty tails.
  inline void check_index_range(long first, long last, long count) {
    if (first < 0 || last < first || last > count)
      throw std::out_of_range("DiagramEvaluator: flat-index range must satisfy 0 <= first <= last <= count");
  }
} // namespace triqs_xca

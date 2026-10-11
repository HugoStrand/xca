// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../../benchmarks/dense_cuda_small/harness.hpp"
#include <gtest/gtest.h>
#include <limits>

TEST(DenseCudaBenchmark, tiny_nonzero_results_cannot_be_replaced_by_zero) {
  dense_cuda_bench::tensor expected(2, 4, 4), actual(2, 4, 4);
  expected = nda::dcomplex{1e-41, -2e-41};
  actual   = 0.;
  EXPECT_THROW(dense_cuda_bench::compare(actual, expected), std::runtime_error);
  actual = expected;
  EXPECT_NO_THROW(dense_cuda_bench::compare(actual, expected));
  actual(0, 0, 0) = {std::numeric_limits<double>::quiet_NaN(), 0.};
  EXPECT_THROW(dense_cuda_bench::compare(actual, expected, false), std::runtime_error);
}
TEST(DenseCudaBenchmark, bounded_selection_covers_long_index_range) {
  constexpr long count = 17179869184L;
  auto x               = dense_cuda_bench::select_indices(count, 8, 1729);
  EXPECT_EQ(x.size(), 8);
  EXPECT_GT(x(7), std::numeric_limits<int>::max());
  for (long i = 0; i < x.size(); ++i) {
    EXPECT_GE(x(i), 0);
    EXPECT_LT(x(i), count);
    if (i) EXPECT_GT(x(i), x(i - 1));
  }
  EXPECT_EQ(dense_cuda_bench::select_indices(2, 8, 1729).size(), 2);
}
TEST(DenseCudaBenchmark, rejects_malformed_integer_arguments) {
  EXPECT_THROW(dense_cuda_bench::integer("4garbage", 2, 16), std::invalid_argument);
  EXPECT_THROW(dense_cuda_bench::integer("0", 1, 12), std::invalid_argument);
}

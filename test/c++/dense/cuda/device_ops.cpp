// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <triqs_xca/dense/cuda/device_ops.hpp>
#ifdef NDA_HAVE_CUDA
#include <nda/gtest_tools.hpp>
#include <cmath>
#include <limits>

namespace {
  using namespace triqs_xca::dense::cuda;
  constexpr auto all = nda::range::all;
  nda::array<complex, 3> values(int r, int N, double phase) {
    nda::array<complex, 3> a(r, N, N);
    for (int t = 0; t < r; ++t)
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) a(t, i, j) = complex{std::sin(.3 * t + .7 * i - .2 * j + phase), std::cos(.2 * t - .1 * i + .8 * j + phase)};
    return a;
  }
} // namespace

TEST(DenseCudaOps, ComplexBatchesAndReusableVertexTables) {
  int r = 5, N = 3;
  DeviceOps ops(r, N);
  auto a = values(r, N, .1), b = values(r, N, .7);
  nda::matrix<complex> f(a(0, all, all));
  tensor ad(a), bd(b), out(r, N, N);
  nda::cumatrix<complex> fd(f);
  auto reference = a;
  ops.batch_product(ad, bd, out);
  for (int t = 0; t < r; ++t) reference(t, all, all) = nda::linalg::matmul(a(t, all, all), b(t, all, all));
  EXPECT_ARRAY_NEAR(reference, nda::to_host(out), 1e-12);
  ops.left_product(fd, bd, out);
  for (int t = 0; t < r; ++t) reference(t, all, all) = nda::linalg::matmul(f, b(t, all, all));
  EXPECT_ARRAY_NEAR(reference, nda::to_host(out), 1e-12);
  ops.right_product(ad, fd, out);
  for (int t = 0; t < r; ++t) reference(t, all, all) = nda::linalg::matmul(a(t, all, all), f);
  EXPECT_ARRAY_NEAR(reference, nda::to_host(out), 1e-12);
  // Rebind the persistent tables to new operands and accumulate with complex scalars.
  complex alpha{.2, -.3}, beta{-.1, .4};
  auto previous = reference;
  ops.left_product(fd, ad, out, alpha, beta);
  for (int t = 0; t < r; ++t) reference(t, all, all) = alpha * nda::linalg::matmul(f, a(t, all, all)) + beta * previous(t, all, all);
  EXPECT_ARRAY_NEAR(reference, nda::to_host(out), 1e-12);
}

TEST(DenseCudaOps, CopiesWeightsAndAccumulation) {
  int r = 4, N = 3;
  DeviceOps ops(r, N);
  auto a = values(r, N, .4);
  tensor ad(a), out(r, N, N), scratch(r, N, N);
  ops.copy(ad, out);
  ops.scale(complex{.5, -.25}, out);
  ops.axpy(complex{-.2, .3}, ad, out);
  auto expected = nda::make_regular(complex{.3, .05} * a);
  EXPECT_ARRAY_NEAR(expected, nda::to_host(out), 1e-13);
  nda::vector<complex> weights(r);
  for (int t = 0; t < r; ++t) weights(t) = complex{.2 * t - .1, .3 * t};
  nda::cuvector<complex> wd(weights);
  ops.weight_rows(ad, wd, scratch);
  for (int t = 0; t < r; ++t) expected(t, all, all) = weights(t) * a(t, all, all);
  EXPECT_ARRAY_NEAR(expected, nda::to_host(scratch), 1e-13);
  ops.zero(out);
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(out))), 0.);
}

TEST(DenseCudaOps, RectangularLayoutRoundTripDoesNotConjugate) {
  DeviceOps ops(5, 3);
  nda::matrix<complex> host(5, 9);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 9; ++j) host(i, j) = complex{.3 * i - j, .7 * j + i};
  nda::cumatrix<complex> cd(host), result(5, 9);
  nda::cumatrix<complex, nda::F_layout> fd(5, 9);
  ops.to_fortran(cd, fd);
  EXPECT_EQ(nda::max_element(nda::abs(host - nda::to_host(fd))), 0.);
  ops.from_fortran(fd, result);
  EXPECT_EQ(nda::max_element(nda::abs(host - nda::to_host(result))), 0.);
}

TEST(DenseCudaOps, NonConjugatingComplexTraces) {
  int r = 4, N = 3, k = 2;
  DeviceOps ops(r, N);
  auto a = values(r, N, .1), operators = values(k, N, .8);
  nda::matrix<complex> packed(k, N * N), expected(r, k);
  for (int l = 0; l < k; ++l)
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) packed(l, i * N + j) = operators(l, j, i);
  for (int t = 0; t < r; ++t)
    for (int l = 0; l < k; ++l) expected(t, l) = nda::trace(nda::linalg::matmul(a(t, all, all), operators(l, all, all)));
  tensor ad(a);
  nda::cumatrix<complex> pd(packed), out(r, k);
  ops.traces(ad, pd, out);
  EXPECT_ARRAY_NEAR(expected, nda::to_host(out), 1e-12);
}

TEST(DenseCudaOps, InvalidLayoutsAliasesAndCheckedSizes) {
  EXPECT_THROW(DeviceOps(0, 3), std::invalid_argument);
  EXPECT_THROW(DeviceOps(3, 50000), std::invalid_argument);
  EXPECT_THROW(DeviceOps::checked_product(std::numeric_limits<std::size_t>::max(), 2), std::overflow_error);
  EXPECT_THROW(DeviceOps::checked_blas_int(static_cast<long>(std::numeric_limits<int>::max()) + 1), std::invalid_argument);
  DeviceOps ops(4, 3);
  tensor a(values(4, 3, .1)), out(4, 3, 3), wrong(3, 3, 3), overlapping(5, 3, 3);
  EXPECT_THROW(ops.batch_product(a, a, a), std::invalid_argument);
  EXPECT_THROW(ops.copy(wrong, out), std::invalid_argument);
  EXPECT_THROW(ops.copy(overlapping(nda::range(0, 4), all, all), overlapping(nda::range(1, 5), all, all)), std::invalid_argument);
  tensor strided(4, 3, 6);
  EXPECT_THROW(ops.copy(strided(all, all, nda::range(0, 6, 2)), out), std::invalid_argument);
  // A failed call does not corrupt workspace or prevent reuse.
  ops.copy(a, out);
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(a) - nda::to_host(out))), 0.);
}
#else
TEST(DenseCudaOps, DISABLED_NoCudaSupport) { GTEST_SKIP() << "CUDA NDA required"; }
#endif

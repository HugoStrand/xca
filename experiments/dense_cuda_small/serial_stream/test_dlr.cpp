// Adapted from frozen test/c++/dense/cuda/dlr_ops.cpp; explicit completion at host assertions.
// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "dlr_ops.hpp"
#ifdef NDA_HAVE_CUDA
#include <cppdlr/cppdlr.hpp>
#include <nda/gtest_tools.hpp>
#include <nda/lapack.hpp>
#include <cmath>
#include <limits>

namespace {
  using namespace xca_experiments::serial_stream;
  constexpr auto all = nda::range::all;
  nda::array<complex, 3> coefficients(int r, int N, double phase) {
    nda::array<complex, 3> result(r, N, N);
    for (int t = 0; t < r; ++t)
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
          result(t, i, j) = complex{std::sin(.7 * t + .3 * i - .2 * j + phase), std::cos(.1 * t - .4 * i + .8 * j + phase)} / (r * N);
    return result;
  }
  cppdlr::imtime_ops small_system() {
    nda::vector<double> rf{-2., 0., 2.}, nodes{0., .4, -.1};
    nda::matrix<double> C(3, 3);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) C(i, j) = cppdlr::k_it(nodes(i), rf(j));
    auto lu = C;
    nda::vector<int> pivots(3);
    if (nda::lapack::getrf(lu, pivots) != 0) throw std::runtime_error("test LU failed");
    return cppdlr::imtime_ops(4., rf, nodes, C, lu, pivots);
  }
} // namespace

TEST(DenseCudaDlr, CachedLUAndInPlaceConversion) {
  auto host = small_system();
  DlrOps device(host, 3);
  auto fc = coefficients(host.rank(), 3, .2), values = host.coefs2vals(fc);
  tensor input(values), output(3, 3, 3), reconstructed(3, 3, 3);
  device.wait_for_default_uploads();
  device.vals2coefs(input, output);
  device.finish(); EXPECT_ARRAY_NEAR(fc, nda::to_host(output), 1e-12);
  device.finish(); EXPECT_ARRAY_NEAR(host.vals2coefs(values), nda::to_host(output), 1e-12);
  device.coefs2vals(output, reconstructed);
  device.finish(); EXPECT_ARRAY_NEAR(values, nda::to_host(reconstructed), 1e-12);
  device.vals2coefs(input, input);
  device.finish(); EXPECT_ARRAY_NEAR(fc, nda::to_host(input), 1e-12);
}

TEST(DenseCudaDlr, RealMeshesReconstructionReflectionAndCoefficientConvolution) {
  for (auto [lambda, eps] : std::vector<std::pair<double, double>>{{10., 1e-10}, {40., 1e-12}, {120., 1e-12}}) {
    for (bool sym : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "lambda=" << lambda << " eps=" << eps << " sym=" << sym);
      cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, eps, sym), sym);
      int r = host.rank(), N = 3;
      DlrOps device(host, N);
      auto fc = coefficients(r, N, .1), gc = coefficients(r, N, .9);
      auto fv = host.coefs2vals(fc), gv = host.coefs2vals(gc);
      tensor fcd(fc), gcd(gc), fvd(fv), out(r, N, N), solved(r, N, N), reconstructed(r, N, N);
      device.wait_for_default_uploads();
      device.vals2coefs(fvd, solved);
      device.coefs2vals(solved, reconstructed);
      device.finish(); EXPECT_ARRAY_NEAR(fv, nda::to_host(reconstructed), 1e-12);
      device.finish(); EXPECT_ARRAY_NEAR(host.coefs2vals(host.vals2coefs(fv)), nda::to_host(reconstructed), 1e-12);
      device.finish(); EXPECT_TRUE(std::isfinite(nda::max_element(nda::abs(nda::to_host(solved)))));
      device.reflect(fvd, out);
      device.finish(); EXPECT_ARRAY_NEAR(host.reflect(fv), nda::to_host(out), 1e-12);
      device.reflect(out, reconstructed);
      device.finish(); EXPECT_ARRAY_NEAR(fv, nda::to_host(reconstructed), 1e-12);
      // This is a matrix convolution, not elementwise multiplication.
      device.finish(); EXPECT_GT(
         nda::max_element(nda::abs(nda::linalg::matmul(fc(0, all, all), gc(0, all, all)) - nda::linalg::matmul(gc(0, all, all), fc(0, all, all)))),
         1e-6);
      for (double beta : {.5, 2., 7.}) {
        device.convolve(beta, fcd, gcd, out);
        device.finish(); EXPECT_ARRAY_NEAR(host.convolve(beta, fc, gc, cppdlr::TIME_ORDERED), nda::to_host(out), 1e-12);
      }
      // Composed edge pipeline exercises the actual evaluator use of both solves.
      tensor gvd(gv), gsolved(r, N, N);
      device.wait_for_default_uploads();
      device.vals2coefs(gvd, gsolved);
      device.convolve(2., solved, gsolved, out);
      auto expected = host.convolve(2., host.vals2coefs(fv), host.vals2coefs(gv), cppdlr::TIME_ORDERED);
      device.finish(); EXPECT_ARRAY_NEAR(expected, nda::to_host(out), 1e-12);
    }
  }
}

TEST(DenseCudaDlr, CancellationInputsBufferReuseAndAliasedConvolution) {
  auto host = small_system();
  DlrOps device(host, 2);
  auto fc = coefficients(3, 2, .3), gc = coefficients(3, 2, .8);
  fc(1, all, all) = -fc(0, all, all) + complex{1e-11, -2e-11} * fc(2, all, all);
  tensor fd(fc), gd(gc), out(3, 2, 2);
  auto expected = host.convolve(3., fc, gc, cppdlr::TIME_ORDERED);
  device.wait_for_default_uploads();
  device.convolve(3., fd, gd, out);
  device.finish(); EXPECT_ARRAY_NEAR(expected, nda::to_host(out), 1e-12);
  device.convolve(3., fd, gd, fd);
  device.finish(); EXPECT_ARRAY_NEAR(expected, nda::to_host(fd), 1e-12);
  // Reupload changed input into the same allocation; scratch must not retain it.
  fc = coefficients(3, 2, 1.3);
  fd = fc;
  device.wait_for_default_uploads();
  device.convolve(.7, fd, gd, gd);
  device.finish(); EXPECT_ARRAY_NEAR(host.convolve(.7, fc, gc, cppdlr::TIME_ORDERED), nda::to_host(gd), 1e-12);
}

#endif

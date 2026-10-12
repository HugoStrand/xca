#include "../serial_stream/dlr_ops.hpp"
#include "dlr_ops.hpp"
#include <cppdlr/cppdlr.hpp>
#include <gtest/gtest.h>
#include <nda/gtest_tools.hpp>
TEST(BatchDlr, WideRhsAndRankTransformsMatchIndependentLanes) {
  using namespace xca_experiments::power_two;
  constexpr auto a = nda::range::all;
  for (double lambda : {10., 40.})
    for (int B : {1, 8, 32, 128}) {
      SCOPED_TRACE(::testing::Message() << "lambda=" << lambda << " B=" << B);
      cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, 1e-12, false), false);
      int r = host.rank(), N = 4;
      DlrOps batched(host, N, B);
      nda::array<complex, 3> coeff(r * B, N, N), second(r * B, N, N), values(r * B, N, N);
      for (int b = 0; b < B; ++b) {
        nda::array<complex, 3> c(r, N, N);
        for (int t = 0; t < r; ++t)
          for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
              c(t, i, j) = complex{std::sin(.3 * t + .7 * i - .2 * j + .01 * b), std::cos(.2 * t - .4 * i + .8 * j + .03 * b)} / (r * N);
        auto v = host.coefs2vals(c);
        for (int t = 0; t < r; ++t)
          for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
              coeff(t * B + b, i, j)  = c(t, i, j);
              values(t * B + b, i, j) = v(t, i, j);
              second(t * B + b, i, j) = complex{.1, .2} * c(t, (i + 1) % N, j);
            }
      }
      tensor cd(coeff), gd(second), vd(values), solved(r * B, N, N), restored(r * B, N, N), out(r * B, N, N);
      batched.wait_for_default_uploads();
      batched.vals2coefs(vd, solved);
      batched.coefs2vals(solved, restored);
      batched.convolve(2., cd, gd, out);
      batched.finish();
      EXPECT_LE(nda::max_element(nda::abs(values - nda::to_host(restored))), 1e-12);
      auto actual = nda::to_host(solved), conv = nda::to_host(out);
      xca_experiments::serial_stream::DlrOps independent(host, N);
      for (int b = 0; b < B; ++b) {
        SCOPED_TRACE(b);
        nda::array<complex, 3> lane(r, N, N), f(r, N, N), g(r, N, N), got(r, N, N), cv(r, N, N);
        for (int t = 0; t < r; ++t)
          for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
              lane(t, i, j) = values(t * B + b, i, j);
              f(t, i, j)    = coeff(t * B + b, i, j);
              g(t, i, j)    = second(t * B + b, i, j);
              got(t, i, j)  = actual(t * B + b, i, j);
              cv(t, i, j)   = conv(t * B + b, i, j);
            }
        tensor lv(lane), lf(f), lg(g), s(r, N, N), c(r, N, N);
        independent.wait_for_default_uploads();
        independent.vals2coefs(lv, s);
        independent.convolve(2., lf, lg, c);
        independent.finish();
        EXPECT_LE(nda::max_element(nda::abs(got - nda::to_host(s))), 1e-12);
        EXPECT_LE(nda::max_element(nda::abs(cv - nda::to_host(c))), 1e-12);
        EXPECT_LE(nda::max_element(nda::abs(cv - host.convolve(2., f, g, cppdlr::TIME_ORDERED))), 1e-12);
      }
    }
}

#include <thread>
TEST(BatchDlr, PendingDefaultStreamUploadIsOrderedBeforeOwnedStream) {
  using namespace xca_experiments::power_two;
  cppdlr::imtime_ops host(10., cppdlr::build_dlr_rf(10., 1e-12, false), false);
  int r = host.rank(), B = 128, N = 4;
  DlrOps batched(host, N, B);
  nda::array<complex, 3> coefficients(r * B, N, N);
  for (int q = 0; q < r * B; ++q)
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) coefficients(q, i, j) = complex{.1 * std::sin(.03 * q + .1 * i), .1 * std::cos(.02 * q + .1 * j)};
  tensor input(r * B, N, N), solved(r * B, N, N), restored(r * B, N, N);
  batched.device_ops().zero(input);
  batched.vals2coefs(input, solved);
  batched.coefs2vals(solved, restored);
  batched.finish();
  void *pinned = nullptr;
  nda::device_error_check(cudaMallocHost(&pinned, coefficients.size() * sizeof(complex)), "test pinned upload");
  struct Guard {
    void *p;
    ~Guard() {
      cudaStreamSynchronize(nullptr);
      cudaFreeHost(p);
    }
  } guard{pinned};
  nda::array<complex, 3> values(r * B, N, N);
  for (int b = 0; b < B; ++b) {
    nda::array<complex, 3> c(r, N, N);
    for (int t = 0; t < r; ++t)
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) c(t, i, j) = coefficients(t * B + b, i, j);
    auto v = host.coefs2vals(c);
    for (int t = 0; t < r; ++t)
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) values(t * B + b, i, j) = v(t, i, j);
  }
  std::memcpy(pinned, values.data(), values.size() * sizeof(complex));
  nda::device_error_check(
     cudaLaunchHostFunc(nullptr, [](void *) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }, nullptr), "test delayed producer");
  nda::device_error_check(cudaMemcpyAsync(input.data(), pinned, coefficients.size() * sizeof(complex), cudaMemcpyHostToDevice, nullptr),
                          "test pending upload");
  batched.wait_for_default_uploads();
  // The delayed producer must complete before the owned stream reads its
  // values.
  batched.vals2coefs(input, solved);
  batched.coefs2vals(solved, restored);
  batched.finish();
  EXPECT_LE(nda::max_element(nda::abs(values - nda::to_host(restored))), 1e-12);
}

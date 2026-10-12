#include "../index_batch/device_ops.hpp"
#include "device_ops.hpp"
#include <gtest/gtest.h>
TEST(FusedPrimitives, WeightedAccumulationAndFinalScale) {
  using namespace xca_experiments::power_two;
  for (int B : {1, 8, 128}) {
    int r = 16, N = 4;
    DeviceOps candidate(r, N, B);
    xca_experiments::index_batch::DeviceOps reference(r, N, B);
    nda::array<complex, 3> h(r * B, N, N), initial(r * B, N, N);
    nda::vector<complex> w(r * B), sign(r * B);
    for (int q = 0; q < r * B; ++q) {
      w(q)    = complex{std::sin(.2 * q), std::cos(.3 * q)};
      sign(q) = q % 2 ? -1. : 1.;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          h(q, i, j)       = complex{std::cos(.2 * q + .3 * i), std::sin(.1 * q - .4 * j)};
          initial(q, i, j) = complex{.1 * i, -.2 * j};
        }
    }
    tensor input(h), actual(initial), expected(initial), scratch(h.shape()), scratch2(h.shape());
    nda::cuvector<complex> weights(w), parity(sign);
    candidate.wait_for_default_uploads();
    reference.wait_for_default_uploads();
    reference.weight_rows(input, weights, scratch);
    reference.weight_rows(scratch, parity, scratch2);
    reference.axpy(1., scratch2, expected);
    reference.finish();
    candidate.weighted_accumulate(input, weights, parity, actual);
    candidate.finish();
    auto a = nda::to_host(actual), e = nda::to_host(expected);
    EXPECT_EQ(nda::max_element(nda::abs(a - e)), 0.);
    reference.axpy(1., input, expected);
    reference.scale(2., expected);
    reference.finish();
    candidate.axpy_scale(input, actual, 2.);
    candidate.finish();
    a = nda::to_host(actual);
    e = nda::to_host(expected);
    EXPECT_LE(nda::max_element(nda::abs(a - e)), 1e-13);
  }
}
TEST(FusedPrimitives, PowerTwoProductsWithComplexAlphaAndBeta) {
  using namespace xca_experiments::power_two;
  if (auto mode = std::getenv("XCA_FUSED_MODE"); !mode || std::string(mode) != "2") GTEST_SKIP() << "mode2 only";
  for (int N : {2, 4, 8, 16, 32, 64}) for (int B : {1, 8, 128}) {
    DeviceOps candidate(16, N, B);
    xca_experiments::index_batch::DeviceOps reference(16, N, B);
    nda::array<complex, 3> h(16 * B, N, N), k(h.shape());
    for (long q = 0; q < h.extent(0); ++q)
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          h(q, i, j) = complex{std::sin(.1 * q + .3 * i - .5 * j), std::cos(.2 * q - .4 * i + .3 * j)};
          k(q, i, j) = complex{std::cos(.3 * q - .2 * i + .1 * j), std::sin(.2 * q + .3 * i - .2 * j)};
        }
    tensor a(h), b(k), actual(h), expected(h);
    candidate.wait_for_default_uploads();
    reference.wait_for_default_uploads();
    reference.batch_product(a, b, expected, complex{.2, .1}, complex{-.1, .3});
    reference.finish();
    candidate.batch_product(a, b, actual, complex{.2, .1}, complex{-.1, .3});
    candidate.finish();
    auto x = nda::to_host(actual), y = nda::to_host(expected);
    EXPECT_LE(nda::max_element(nda::abs(x - y)), 1e-10 * nda::max_element(nda::abs(y)) + 1e-280);
    // Lane operators differ and both directions must preserve pointer-table order.
    for (bool left : {false, true}) for (complex beta : {complex{0., 0.}, complex{1., 0.}}) {
      actual = h;
      expected = h;
      std::vector<complex const *> operators;
      for (int lane = 0; lane < B; ++lane) operators.push_back(b.data() + lane * N * N);
      DeviceOps::LaneTables candidate_tables(16 * B);
      xca_experiments::index_batch::DeviceOps::LaneTables reference_tables(16 * B);
      candidate.prepare_lanes(candidate_tables, operators, a, actual, left);
      reference.prepare_lanes(reference_tables, operators, a, expected, left);
      candidate.wait_for_default_uploads();
      reference.wait_for_default_uploads();
      reference.lane_vertex(reference_tables, complex{.2, .1}, beta);
      reference.finish();
      candidate.lane_vertex(candidate_tables, complex{.2, .1}, beta);
      candidate.finish();
      x = nda::to_host(actual);
      y = nda::to_host(expected);
      double norm = 0., error = 0.;
      for (long q = 0; q < x.extent(0); ++q) for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        ASSERT_TRUE(std::isfinite(std::abs(x(q, i, j))));
        ASSERT_TRUE(std::isfinite(std::abs(y(q, i, j))));
        norm = std::max(norm, std::abs(y(q, i, j)));
        error = std::max(error, std::abs(x(q, i, j) - y(q, i, j)));
      }
      EXPECT_LE(error, 1e-280 + 1e-10 * norm) << "N=" << N << " B=" << B << " left=" << left;
    }
  }
}

#include <cppdlr/fast_convolve.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>

namespace {

  using nda::dcomplex;
  using nda::range;

  cppdlr::imtime_ops const &itops() {
    static auto const ops = cppdlr::imtime_ops(1e5, cppdlr::build_dlr_rf(1e5, 1e-12));
    return ops;
  }

  double rate(long batch, bool second) { return 0.3 + 0.17 * batch + (second ? 0.11 * batch : 0.); }

  dcomplex amplitude(long batch, long i, long j, bool second) {
    return {(1. + 0.2 * i - 0.13 * j + 0.3 * batch) / (1. + i + j), (0.1 * i + 0.23 * j - 0.31 * batch + 0.27 * second) / (1. + i)};
  }

  void fill_values(nda::array_view<dcomplex, 4> values, bool second) {
    auto const &ops = itops();
    for (long l = 0; l < values.shape(0); ++l) {
      double t = ops.get_itnodes(l);
      if (t < 0.) t += 1.;
      for (long b = 0; b < values.shape(1); ++b)
        for (long i = 0; i < values.shape(2); ++i)
          for (long j = 0; j < values.shape(3); ++j) values(l, b, i, j) = amplitude(b, i, j, second) * std::exp(-rate(b, second) * t);
    }
  }

  void expect_convolution(double beta, nda::array_const_view<dcomplex, 4> f, nda::array_const_view<dcomplex, 4> g,
                          nda::array_const_view<dcomplex, 4> actual) {
    auto const &ops = itops();
    for (long b = 0; b < f.shape(1); ++b) {
      SCOPED_TRACE(b);
      // A lane is strided along the DLR axis; explicit reference copies are test-only.
      auto f_lane    = nda::make_regular(f(range::all, b, range::all, range::all));
      auto g_lane    = nda::make_regular(g(range::all, b, range::all, range::all));
      auto reference = ops.convolve(beta, ops.vals2coefs(f_lane), ops.vals2coefs(g_lane), cppdlr::TIME_ORDERED);
      double scale   = std::max(1., nda::max_element(nda::abs(reference)));
      EXPECT_LE(nda::max_element(nda::abs(actual(range::all, b, range::all, range::all) - reference)), 2e-10 * scale);

      // Each batch member has its own noncommuting pair, including equal rates at b=0.
      for (long l = 0; l < ops.rank(); ++l) {
        double t = ops.get_itnodes(l);
        if (t < 0.) t += 1.;
        double integral = b == 0 ? t * std::exp(-rate(b, false) * t) :
                                   (std::exp(-rate(b, true) * t) - std::exp(-rate(b, false) * t)) / (rate(b, false) - rate(b, true));
        for (long i = 0; i < f.shape(2); ++i)
          for (long j = 0; j < g.shape(3); ++j) {
            dcomplex expected = 0.;
            for (long k = 0; k < f.shape(3); ++k) expected += beta * integral * amplitude(b, i, k, false) * amplitude(b, k, j, true);
            EXPECT_LE(std::abs(actual(l, b, i, j) - expected), 2e-10 * scale);
          }
      }
    }
  }

} // namespace

TEST(FastConvolveBatched, rectangular_products_match_reference_and_analytic_integral) {
  auto const &ops = itops();
  ASSERT_GE(ops.rank(), 80);
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_batched_workspace work(ops.rank(), 4, 2, 3, 4);
  nda::array<dcomplex, 4> f(ops.rank(), 4, 2, 3), g(ops.rank(), 4, 3, 4), result(ops.rank(), 4, 2, 4);
  fill_values(f, false);
  fill_values(g, true);
  result() = dcomplex(17., -9.);
  convolution.convolve(2.3, f, g, result, work);
  expect_convolution(2.3, f, g, result);
}

TEST(FastConvolveBatched, changing_active_batch_reuses_capacity_and_preserves_tail) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_batched_workspace work(ops.rank(), 5, 2, 3, 4);
  nda::array<dcomplex, 4> f_storage(ops.rank(), 5, 2, 3), g_storage(ops.rank(), 5, 3, 4), out_storage(ops.rank(), 5, 2, 4);
  auto const *fc_data      = work.fc.data();
  auto const *gc_data      = work.gc.data();
  auto const *hfc_data     = work.hfc.data();
  auto const *hgc_data     = work.hgc.data();
  auto const *product_data = work.product.data();
  std::array owners{&work.fc, &work.gc, &work.hfc, &work.hgc, &work.product};
  dcomplex const sentinel(29., -17.);

  for (long batch : {5, 2, 1, 4, 5}) {
    SCOPED_TRACE(batch);
    nda::array_view<dcomplex, 4> f{{ops.rank(), batch, 2, 3}, f_storage.data()};
    nda::array_view<dcomplex, 4> g{{ops.rank(), batch, 3, 4}, g_storage.data()};
    nda::array_view<dcomplex, 4> result{{ops.rank(), batch, 2, 4}, out_storage.data()};
    fill_values(f, false);
    fill_values(g, true);
    for (auto *owner : owners) (*owner)() = sentinel;
    out_storage() = sentinel;
    convolution.convolve(1.7, f, g, result, work);
    expect_convolution(1.7, f, g, result);
    for (long index = result.size(); index < out_storage.size(); ++index) EXPECT_EQ(out_storage.data()[index], sentinel);
    std::array active_sizes{f.size(), g.size(), f.size(), g.size(), result.size()};
    for (std::size_t i = 0; i < owners.size(); ++i)
      for (long index = active_sizes[i]; index < owners[i]->size(); ++index) EXPECT_EQ(owners[i]->data()[index], sentinel);
    EXPECT_EQ(work.fc.data(), fc_data);
    EXPECT_EQ(work.gc.data(), gc_data);
    EXPECT_EQ(work.hfc.data(), hfc_data);
    EXPECT_EQ(work.hgc.data(), hgc_data);
    EXPECT_EQ(work.product.data(), product_data);
  }
}

TEST(FastConvolveBatched, output_can_replace_either_input) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_batched_workspace work(ops.rank(), 5, 3, 3, 3);
  nda::array<dcomplex, 4> f(ops.rank(), 3, 3, 3), g(f.shape());
  fill_values(f, false);
  fill_values(g, true);
  auto overwritten_f = f;
  convolution.convolve(2.1, overwritten_f, g, overwritten_f, work);
  expect_convolution(2.1, f, g, overwritten_f);
  auto overwritten_g = g;
  convolution.convolve(2.1, f, overwritten_g, overwritten_g, work);
  expect_convolution(2.1, f, g, overwritten_g);
}

TEST(FastConvolveBatched, coefficient_solve_reconstructs_all_batch_members) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  nda::array<dcomplex, 4> values(ops.rank(), 4, 2, 3), coefficients(values.shape());
  fill_values(values, false);
  convolution.vals2coefs(values, coefficients);
  auto expected = nda::reshape(values, ops.rank(), 8, 3);
  auto actual   = ops.coefs2vals(nda::reshape(coefficients, ops.rank(), 8, 3));
  EXPECT_LE(nda::max_element(nda::abs(actual - expected)), 2e-12 * nda::max_element(nda::abs(expected)));

  auto overwritten_values = values;
  convolution.vals2coefs(overwritten_values, overwritten_values);
  auto reconstructed = ops.coefs2vals(nda::reshape(overwritten_values, ops.rank(), 8, 3));
  EXPECT_LE(nda::max_element(nda::abs(reconstructed - expected)), 2e-12 * nda::max_element(nda::abs(expected)));
}

TEST(FastConvolveBatched, invalid_shapes_and_capacity_are_rejected) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_batched_workspace work(ops.rank(), 4, 2, 3, 4), small_work(ops.rank(), 2, 2, 3, 4);
  cppdlr::fast_convolve_batched_workspace wrong_work(ops.rank(), 4, 3, 2, 4);
  nda::array<dcomplex, 4> f(ops.rank(), 4, 2, 3), g(ops.rank(), 4, 3, 4), result(ops.rank(), 4, 2, 4);
  nda::array<dcomplex, 4> wrong_rank(ops.rank() - 1, 4, 2, 3), wrong_batch(ops.rank(), 3, 3, 4);
  nda::array<dcomplex, 4> wrong_inner(ops.rank(), 4, 2, 4), wrong_output(ops.rank(), 4, 2, 3);
  nda::array<dcomplex, 4> zero_batch(ops.rank(), 0, 2, 3);
  fill_values(f, false);
  fill_values(g, true);

  EXPECT_ANY_THROW(convolution.vals2coefs(f, g));
  EXPECT_ANY_THROW(convolution.convolve(1., wrong_rank, g, result, work));
  EXPECT_ANY_THROW(convolution.convolve(1., f, wrong_batch, result, work));
  EXPECT_ANY_THROW(convolution.convolve(1., f, wrong_inner, result, work));
  EXPECT_ANY_THROW(convolution.convolve(1., f, g, wrong_output, work));
  EXPECT_ANY_THROW(convolution.convolve(1., f, g, result, small_work));
  EXPECT_ANY_THROW(convolution.convolve(1., f, g, result, wrong_work));
  EXPECT_ANY_THROW(convolution.vals2coefs(zero_batch, zero_batch));
  EXPECT_ANY_THROW(cppdlr::fast_convolve_batched_workspace(ops.rank(), 0, 2, 3, 4));
  EXPECT_ANY_THROW(cppdlr::fast_convolve_batched_workspace(ops.rank(), 4, 0, 3, 4));
  EXPECT_ANY_THROW(cppdlr::fast_convolve_batched_workspace(ops.rank(), std::numeric_limits<int>::max(), 2, 3, 4));
}

TEST(FastConvolveBatched, strided_batch_slices_and_invalid_beta_are_rejected) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_batched_workspace work(ops.rank(), 4, 2, 3, 4);
  nda::array<dcomplex, 4> f_storage(ops.rank(), 5, 2, 3), g_storage(ops.rank(), 5, 3, 4), out_storage(ops.rank(), 5, 2, 4);
  nda::array<dcomplex, 4> f(ops.rank(), 4, 2, 3), g(ops.rank(), 4, 3, 4), result(ops.rank(), 4, 2, 4);
  auto strided_f      = f_storage(range::all, range(0, 4), range::all, range::all);
  auto strided_g      = g_storage(range::all, range(0, 4), range::all, range::all);
  auto strided_output = out_storage(range::all, range(0, 4), range::all, range::all);
  EXPECT_ANY_THROW(convolution.vals2coefs(strided_f, f));
  EXPECT_ANY_THROW(convolution.vals2coefs(f, strided_f));
  EXPECT_ANY_THROW(convolution.convolve(1., strided_f, g, result, work));
  EXPECT_ANY_THROW(convolution.convolve(1., f, strided_g, result, work));
  EXPECT_ANY_THROW(convolution.convolve(1., f, g, strided_output, work));
  for (double beta : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    EXPECT_ANY_THROW(convolution.convolve(beta, f, g, result, work));
}

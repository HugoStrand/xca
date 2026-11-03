#include <cppdlr/fast_convolve.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>

namespace {

  using nda::dcomplex;
  using tensor = nda::array<dcomplex, 3>;

  cppdlr::imtime_ops const &itops() {
    // A DLR rank around 100 exercises the ill-conditioned interpolation solve.
    static auto const ops = cppdlr::imtime_ops(1e5, cppdlr::build_dlr_rf(1e5, 1e-12));
    return ops;
  }

  std::array<double, 3> rates(int input) { return input == 0 ? std::array{0.2, 1.3, 4.7} : std::array{0.2, 2.1, 3.6}; }

  dcomplex amplitude(int term, long i, long j, int input, bool complex) {
    double real = (1. + 0.3 * i - 0.2 * j + 0.17 * input) / (1. + term + i + 2. * j);
    double imag = complex ? (0.4 * i - 0.3 * j + 0.23 * input - 0.19 * term) / (1. + i + j) : 0.;
    return {real, imag};
  }

  tensor values(long rows, long cols, int input, bool complex) {
    auto const &ops = itops();
    auto result     = tensor::zeros(ops.rank(), rows, cols);
    auto exponents  = rates(input);
    for (long l = 0; l < ops.rank(); ++l) {
      double t = ops.get_itnodes()(l);
      if (t < 0) t += 1.;
      for (long i = 0; i < rows; ++i)
        for (long j = 0; j < cols; ++j)
          for (int k = 0; k < 3; ++k) result(l, i, j) += amplitude(k, i, j, input, complex) * std::exp(-exponents[k] * t);
    }
    return result;
  }

  tensor analytic_convolution(double beta, long rows, long inner, long cols, bool complex) {
    auto const &ops = itops();
    auto result     = tensor::zeros(ops.rank(), rows, cols);
    auto f_rates    = rates(0);
    auto g_rates    = rates(1);
    for (long l = 0; l < ops.rank(); ++l) {
      double t = ops.get_itnodes()(l);
      if (t < 0) t += 1.;
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          double integral = f_rates[a] == g_rates[b] ? t * std::exp(-f_rates[a] * t) :
                                                       (std::exp(-g_rates[b] * t) - std::exp(-f_rates[a] * t)) / (f_rates[a] - g_rates[b]);
          for (long i = 0; i < rows; ++i)
            for (long j = 0; j < cols; ++j)
              for (long k = 0; k < inner; ++k) result(l, i, j) += beta * integral * amplitude(a, i, k, 0, complex) * amplitude(b, k, j, 1, complex);
        }
    }
    return result;
  }

  void expect_close(tensor const &actual, tensor const &expected, double tolerance = 2e-10) {
    ASSERT_EQ(actual.shape(), expected.shape());
    double scale = std::max(1., nda::max_element(nda::abs(expected)));
    EXPECT_LE(nda::max_element(nda::abs(actual - expected)), tolerance * scale);
  }

} // namespace

TEST(FastConvolve, coefficient_solve_reconstructs_values) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  ASSERT_GE(ops.rank(), 80);

  for (bool complex : {false, true}) {
    SCOPED_TRACE(complex ? "complex" : "real-valued");
    auto input = values(2, 3, 0, complex);
    tensor coefficients(input.shape());
    convolution.vals2coefs(input, coefficients);
    auto reconstructed = ops.coefs2vals(coefficients);
    expect_close(reconstructed, input, 2e-12);

    // Coefficients themselves are sensitive to the nearly singular DLR basis;
    // compare the represented functions instead of imposing coefficient equality.
    auto reference = ops.coefs2vals(ops.vals2coefs(input));
    expect_close(reconstructed, reference, 2e-12);

    auto overwritten_input = input;
    convolution.vals2coefs(overwritten_input, overwritten_input);
    expect_close(ops.coefs2vals(overwritten_input), input, 2e-12);
  }
}

TEST(FastConvolve, rectangular_products_match_reference_and_analytic_integral) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 2, 3, 4);
  tensor result(ops.rank(), 2, 4);
  double const beta = 2.3;

  for (bool complex : {false, true}) {
    SCOPED_TRACE(complex ? "complex" : "real-valued");
    auto f         = values(2, 3, 0, complex);
    auto g         = values(3, 4, 1, complex);
    auto reference = ops.convolve(beta, ops.vals2coefs(f), ops.vals2coefs(g), cppdlr::TIME_ORDERED);
    result()       = dcomplex(17., -9.);
    convolution.convolve(beta, f, g, result, workspace);
    expect_close(result, reference);
    expect_close(result, analytic_convolution(beta, 2, 3, 4, complex));

    // Reusing output and scratch must replace previous results and consume the new inputs.
    f *= dcomplex(0.7, -0.2);
    g *= dcomplex(-0.4, 0.3);
    convolution.convolve(beta, f, g, result, workspace);
    expect_close(result, nda::make_regular(dcomplex(0.7, -0.2) * dcomplex(-0.4, 0.3) * reference));
  }
}

TEST(FastConvolve, output_can_replace_either_input) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 3, 3, 3);
  auto f         = values(3, 3, 0, true);
  auto g         = values(3, 3, 1, true);
  auto reference = ops.convolve(1.7, ops.vals2coefs(f), ops.vals2coefs(g), cppdlr::TIME_ORDERED);

  auto overwritten_f = f;
  convolution.convolve(1.7, overwritten_f, g, overwritten_f, workspace);
  expect_close(overwritten_f, reference);
  auto overwritten_g = g;
  convolution.convolve(1.7, f, overwritten_g, overwritten_g, workspace);
  expect_close(overwritten_g, reference);
}

TEST(FastConvolve, signed_kernel_mixture_resolves_boundary_layers) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 1, 1, 1);
  std::array frequencies{-0.8 * ops.lambda(), -3.7, 0., 1.4, 0.8 * ops.lambda()};
  double const beta = 3.7;
  auto f            = tensor::zeros(ops.rank(), 1, 1);
  auto g            = tensor::zeros(ops.rank(), 1, 1);
  auto expected     = tensor::zeros(ops.rank(), 1, 1);
  tensor result(ops.rank(), 1, 1);

  // Stable DLR kernels avoid overflow for the strongly negative frequencies.
  // The equal-frequency integral covers zero and both signs; opposite extremes
  // create narrow boundary layers near opposite ends of the time interval.
  for (long l = 0; l < ops.rank(); ++l) {
    double const t_relative = ops.get_itnodes(l);
    double const t          = t_relative < 0 ? 1. + t_relative : t_relative;
    for (int a = 0; a < int(frequencies.size()); ++a) {
      double const ka = cppdlr::k_it(t_relative, frequencies[a]);
      f(l, 0, 0) += amplitude(a, 0, 0, 0, true) * ka;
      g(l, 0, 0) += amplitude(a, 0, 0, 1, true) * ka;
      for (int b = 0; b < int(frequencies.size()); ++b) {
        double const k0b      = cppdlr::k_it(0., frequencies[b]);
        double const integral = a == b ?
           t * ka * k0b :
           (ka * k0b - cppdlr::k_it(t_relative, frequencies[b]) * cppdlr::k_it(0., frequencies[a])) / (frequencies[b] - frequencies[a]);
        expected(l, 0, 0) += beta * amplitude(a, 0, 0, 0, true) * amplitude(b, 0, 0, 1, true) * integral;
      }
    }
  }
  convolution.convolve(beta, f, g, result, workspace);
  expect_close(result, expected);
  expect_close(result, ops.convolve(beta, ops.vals2coefs(f), ops.vals2coefs(g), cppdlr::TIME_ORDERED));
}

TEST(FastConvolve, incompatible_shapes_are_rejected) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 2, 3, 4);
  auto f = values(2, 3, 0, true);
  auto g = values(3, 4, 1, true);
  tensor result(ops.rank(), 2, 4);
  tensor wrong_rank(ops.rank() - 1, 2, 3);
  tensor wrong_coefficients(ops.rank(), 3, 2);
  tensor wrong_inner(ops.rank(), 2, 4);
  tensor wrong_output(ops.rank(), 2, 3);
  cppdlr::fast_convolve_workspace wrong_workspace(ops.rank(), 2, 3, 3);

  EXPECT_ANY_THROW(convolution.vals2coefs(wrong_rank, wrong_rank));
  EXPECT_ANY_THROW(convolution.vals2coefs(f, wrong_coefficients));
  EXPECT_ANY_THROW(convolution.convolve(1., wrong_rank, g, result, workspace));
  EXPECT_ANY_THROW(convolution.convolve(1., f, wrong_inner, result, workspace));
  EXPECT_ANY_THROW(convolution.convolve(1., f, g, wrong_output, workspace));
  EXPECT_ANY_THROW(convolution.convolve(1., f, g, result, wrong_workspace));
}

TEST(FastConvolve, noncontiguous_views_and_invalid_beta_are_rejected) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 2, 3, 4);
  auto f = values(2, 3, 0, true);
  auto g = values(3, 4, 1, true);
  tensor result(ops.rank(), 2, 4);
  tensor strided_f_storage(ops.rank(), 2, 6);
  tensor strided_g_storage(ops.rank(), 3, 8);
  tensor strided_output_storage(ops.rank(), 2, 8);
  auto strided_f      = strided_f_storage(nda::range::all, nda::range::all, nda::range(0, 6, 2));
  auto strided_g      = strided_g_storage(nda::range::all, nda::range::all, nda::range(0, 8, 2));
  auto strided_output = strided_output_storage(nda::range::all, nda::range::all, nda::range(0, 8, 2));

  EXPECT_ANY_THROW(convolution.vals2coefs(strided_f, workspace.fc));
  EXPECT_ANY_THROW(convolution.vals2coefs(f, strided_f));
  EXPECT_ANY_THROW(convolution.convolve(1., strided_f, g, result, workspace));
  EXPECT_ANY_THROW(convolution.convolve(1., f, strided_g, result, workspace));
  EXPECT_ANY_THROW(convolution.convolve(1., f, g, strided_output, workspace));
  for (double beta : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    EXPECT_ANY_THROW(convolution.convolve(beta, f, g, result, workspace));
}

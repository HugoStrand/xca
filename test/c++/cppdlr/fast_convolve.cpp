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

TEST(FastConvolve, prepared_operand_can_be_reused_on_either_side) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  auto cached_values = values(2, 3, 0, true);
  tensor cached_coefficients(cached_values.shape()), cached_hilbert(cached_values.shape());
  convolution.prepare_operand(cached_values, cached_coefficients, cached_hilbert);
  auto const saved_coefficients = cached_coefficients;
  auto const saved_hilbert      = cached_hilbert;
  tensor output_storage(ops.rank(), 4, 4), product_storage(ops.rank(), 4, 4);
  dcomplex const sentinel(29., -17.);
  bool saw_overwrite = false, saw_accumulate = false;
  auto multiply = [&](nda::matrix_const_view<dcomplex> a, nda::matrix_const_view<dcomplex> b, nda::matrix_view<dcomplex> c, double accumulate) {
    EXPECT_EQ(a.shape(1), b.shape(0));
    EXPECT_EQ(c.shape(0), a.shape(0));
    EXPECT_EQ(c.shape(1), b.shape(1));
    EXPECT_TRUE(accumulate == 0. || accumulate == 1.);
    saw_overwrite |= accumulate == 0.;
    saw_accumulate |= accumulate == 1.;
    nda::blas::gemm(1., a, b, accumulate, c);
  };

  // Reuse one rectangular operand both as the left and as the right factor.
  // The distinct complex operands make accidentally reversing products visible.
  for (bool cached_left : {true, false, true}) {
    SCOPED_TRACE(cached_left ? "cached left operand" : "cached right operand");
    auto other = cached_left ? values(3, 4, 1, true) : values(4, 2, 1, true);
    tensor coefficients(other.shape()), hilbert(other.shape());
    convolution.prepare_operand(other, coefficients, hilbert);
    auto const &f   = cached_left ? cached_values : other;
    auto const &g   = cached_left ? other : cached_values;
    auto const &fc  = cached_left ? cached_coefficients : coefficients;
    auto const &gc  = cached_left ? coefficients : cached_coefficients;
    auto const &hfc = cached_left ? cached_hilbert : hilbert;
    auto const &hgc = cached_left ? hilbert : cached_hilbert;
    auto reference = ops.convolve(2.3, ops.vals2coefs(f), ops.vals2coefs(g), cppdlr::TIME_ORDERED);
    nda::array_view<dcomplex, 3> result{{ops.rank(), f.shape(1), g.shape(2)}, output_storage.data()};
    nda::array_view<dcomplex, 3> product{result.shape(), product_storage.data()};
    output_storage()  = sentinel;
    product_storage() = sentinel;
    convolution.convolve_prepared(2.3, fc, gc, hfc, hgc, result, product, multiply);
    expect_close(nda::make_regular(result), reference);
    EXPECT_EQ(nda::max_element(nda::abs(cached_coefficients - saved_coefficients)), 0.);
    EXPECT_EQ(nda::max_element(nda::abs(cached_hilbert - saved_hilbert)), 0.);
    for (long i = result.size(); i < output_storage.size(); ++i) EXPECT_EQ(output_storage.data()[i], sentinel);
    for (long i = product.size(); i < product_storage.size(); ++i) EXPECT_EQ(product_storage.data()[i], sentinel);
  }
  EXPECT_TRUE(saw_overwrite);
  EXPECT_TRUE(saw_accumulate);
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

TEST(FastConvolve, changing_rectangular_shapes_reuse_capacity_and_preserve_tails) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 4, 4, 4);
  std::array owners{&workspace.fc, &workspace.gc, &workspace.hfc, &workspace.hgc, &workspace.product};
  std::array<dcomplex const *, 5> addresses;
  for (std::size_t i = 0; i < owners.size(); ++i) addresses[i] = owners[i]->data();
  tensor output_storage(ops.rank(), 4, 4);
  dcomplex const sentinel(29., -17.);
  std::array shapes{std::array{4L, 4L, 4L}, std::array{1L, 1L, 1L}, std::array{2L, 3L, 4L},
                    std::array{3L, 4L, 2L}, std::array{4L, 2L, 3L}, std::array{4L, 4L, 4L}};

  for (auto [m, n, p] : shapes) {
    SCOPED_TRACE(::testing::Message() << m << "x" << n << "x" << p);
    auto f = values(m, n, 0, true);
    auto g = values(n, p, 1, true);
    auto reference = ops.convolve(2.3, ops.vals2coefs(f), ops.vals2coefs(g), cppdlr::TIME_ORDERED);
    nda::array_view<dcomplex, 3> result{{ops.rank(), m, p}, output_storage.data()};
    for (auto *owner : owners) (*owner)() = sentinel;
    output_storage() = sentinel;

    convolution.convolve(2.3, f, g, result, workspace);
    expect_close(nda::make_regular(result), reference);
    expect_close(nda::make_regular(result), analytic_convolution(2.3, m, n, p, true));
    std::array active_sizes{f.size(), g.size(), f.size(), g.size(), result.size()};
    for (std::size_t i = 0; i < owners.size(); ++i) {
      EXPECT_EQ(owners[i]->data(), addresses[i]);
      EXPECT_EQ(owners[i]->shape(), (std::array<long, 3>{ops.rank(), 4, 4}));
      for (long j = active_sizes[i]; j < owners[i]->size(); ++j) EXPECT_EQ(owners[i]->data()[j], sentinel);
    }
    for (long j = result.size(); j < output_storage.size(); ++j) EXPECT_EQ(output_storage.data()[j], sentinel);
  }
}

TEST(FastConvolve, changing_output_shape_can_overwrite_either_packed_input) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 4, 4, 4);
  tensor f_storage(ops.rank(), 4, 4), g_storage(ops.rank(), 4, 4);
  dcomplex const sentinel(31., -11.);
  std::array shapes{std::array{2L, 1L, 4L}, std::array{2L, 4L, 1L}, std::array{4L, 1L, 2L}, std::array{1L, 4L, 2L}};

  for (auto [m, n, p] : shapes) {
    auto f_values = values(m, n, 0, true);
    auto g_values = values(n, p, 1, true);
    auto reference = ops.convolve(1.7, ops.vals2coefs(f_values), ops.vals2coefs(g_values), cppdlr::TIME_ORDERED);
    for (bool replace_f : {true, false}) {
      SCOPED_TRACE(::testing::Message() << m << "x" << n << "x" << p << ", replace " << (replace_f ? "f" : "g"));
      f_storage() = sentinel;
      g_storage() = sentinel;
      nda::array_view<dcomplex, 3> f{{ops.rank(), m, n}, f_storage.data()};
      nda::array_view<dcomplex, 3> g{{ops.rank(), n, p}, g_storage.data()};
      f = f_values;
      g = g_values;
      auto &aliased_storage = replace_f ? f_storage : g_storage;
      nda::array_view<dcomplex, 3> result{{ops.rank(), m, p}, aliased_storage.data()};
      convolution.convolve(1.7, f, g, result, workspace);
      expect_close(nda::make_regular(result), reference);
      expect_close(nda::make_regular(result), analytic_convolution(1.7, m, n, p, true));
      if (replace_f) EXPECT_EQ(nda::max_element(nda::abs(g - g_values)), 0.);
      else EXPECT_EQ(nda::max_element(nda::abs(f - f_values)), 0.);
      long const used = std::max(result.size(), replace_f ? f.size() : g.size());
      for (long j = used; j < aliased_storage.size(); ++j) EXPECT_EQ(aliased_storage.data()[j], sentinel);
    }
  }
}

TEST(FastConvolve, signed_kernel_mixture_resolves_boundary_layers) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  cppdlr::fast_convolve_workspace workspace(ops.rank(), 4, 4, 4);
  // Shrink to a scalar after a larger rectangular call has populated scratch.
  tensor warmup(ops.rank(), 4, 2);
  convolution.convolve(1., values(4, 3, 0, true), values(3, 2, 1, true), warmup, workspace);
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

TEST(FastConvolve, workspace_capacity_and_allocation_limits_are_checked) {
  auto const &ops = itops();
  cppdlr::fast_convolve convolution(ops);
  auto f = values(2, 3, 0, true);
  auto g = values(3, 4, 1, true);
  tensor result(ops.rank(), 2, 4);
  std::array entries{6L, 12L, 6L, 12L, 8L};
  for (std::size_t i = 0; i < entries.size(); ++i) {
    SCOPED_TRACE(i);
    cppdlr::fast_convolve_workspace workspace(ops.rank(), 4, 4, 4);
    std::array owners{&workspace.fc, &workspace.gc, &workspace.hfc, &workspace.hgc, &workspace.product};
    owners[i]->resize(ops.rank(), 1, entries[i] - 1);
    EXPECT_THROW(convolution.convolve(1., f, g, result, workspace), std::runtime_error);
  }
  cppdlr::fast_convolve_workspace wrong_rank(ops.rank() + 1, 4, 4, 4);
  EXPECT_THROW(convolution.convolve(1., f, g, result, wrong_rank), std::runtime_error);
  EXPECT_THROW(cppdlr::fast_convolve_workspace(0, 1, 1, 1), std::runtime_error);
  EXPECT_THROW(cppdlr::fast_convolve_workspace(ops.rank(), 1, -1, 1), std::runtime_error);
  EXPECT_THROW(cppdlr::fast_convolve_workspace(std::numeric_limits<long>::max(), 1, 1, 1), std::runtime_error);
  EXPECT_THROW(cppdlr::fast_convolve_workspace(ops.rank(), std::numeric_limits<int>::max(), 2, 1), std::runtime_error);
  // Every BLAS dimension fits, but the complete allocation cannot fit in bytes.
  EXPECT_THROW(cppdlr::fast_convolve_workspace(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 1, 1), std::runtime_error);
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

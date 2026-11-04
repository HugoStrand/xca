#pragma once

// Isolated CPU experiment, not a production API. The complex_blas policy mirrors
// fast_convolve; other policies change one arithmetic implementation at a time.
#include <cppdlr/fast_convolve.hpp>
#include <array>
#include <chrono>
#include <complex>

namespace small_n {
  using Complex = nda::dcomplex;
  using Input = nda::array_const_view<Complex, 4>;
  using Output = nda::array_view<Complex, 4>;
  enum class Policy { complex_blas, fixed4, real_transform, real_transform_fixed4, real_all, real_all_fixed4 };
  inline constexpr std::array policies{Policy::complex_blas, Policy::fixed4, Policy::real_transform, Policy::real_transform_fixed4,
                                       Policy::real_all, Policy::real_all_fixed4};
  inline char const *name(Policy p) {
    switch (p) {
      case Policy::complex_blas: return "complex_blas";
      case Policy::fixed4: return "fixed4";
      case Policy::real_transform: return "real_transform";
      case Policy::real_transform_fixed4: return "real_transform_fixed4";
      case Policy::real_all: return "real_all";
      case Policy::real_all_fixed4: return "real_all_fixed4";
    }
    throw std::runtime_error("unknown convolution experiment policy");
  }

  // Non-conjugated 4x4 GEMM for finite scientific inputs, without fast-math.
  // Accept arbitrary matrix strides; output must not overlap either input.
  // The beta==0 branch deliberately does not read the old output.
  template <typename A, typename B, typename C> inline void gemm4(Complex alpha, A const &a, B const &b, Complex beta, C &&c) {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        double real = 0., imag = 0.;
        for (int k = 0; k < 4; ++k) {
          Complex av = a(i, k), bv = b(k, j);
          real += av.real() * bv.real() - av.imag() * bv.imag();
          imag += av.real() * bv.imag() + av.imag() * bv.real();
        }
        Complex sum{real, imag};
        c(i, j) = beta == Complex{} ? alpha * sum : alpha * sum + beta * c(i, j);
      }
  }

  struct Stages {
    // Aggregate durations in seconds, not percentages of an uninstrumented run.
    std::array<double, 7> seconds{};
    static constexpr std::array<char const *, 7> names{
       "values_to_coefficients", "hilbert_transforms", "diagonal_products", "diagonal_reconstruction", "cross_products",
       "cross_reconstruction", "scale"};
  };

  class Convolution {
    cppdlr::imtime_ops const &itops;
    cppdlr::fast_convolve production;
    nda::matrix<Complex> cf2it, hilb, tcf2it;
    nda::matrix<double> real_cf2it, real_hilb, real_tcf2it;

    static void check_array(Input a, long rank) {
      if (a.shape(0) != rank || a.shape(1) <= 0 || a.shape(2) <= 0 || a.shape(3) <= 0)
        throw std::runtime_error("convolution experiment: invalid shape");
      if (!a.indexmap().is_contiguous() || !a.indexmap().has_positive_strides())
        throw std::runtime_error("convolution experiment: expected contiguous C-layout");
      if (a.shape(1) > std::numeric_limits<int>::max() / a.shape(2)
          || a.shape(1) * a.shape(2) > std::numeric_limits<int>::max() / a.shape(3) / 2)
        throw std::runtime_error("convolution experiment: flattened real dimension exceeds BLAS integer range");
    }

    static void transform(nda::matrix_const_view<Complex> a, Input b, Output c, double accumulate) {
      auto bf = nda::reshape(b, b.shape(0), b.shape(1) * b.shape(2) * b.shape(3));
      auto cf = nda::reshape(c, c.shape(0), c.shape(1) * c.shape(2) * c.shape(3));
      nda::blas::gemm(1., a, bf, accumulate, cf);
    }

    static void real_transform(nda::matrix_const_view<double> a, Input b, Output c, double accumulate) {
      static_assert(sizeof(Complex) == 2 * sizeof(double));
      // Real and imaginary parts are independent RHS columns of a real matrix.
      // std::complex<double>'s array-oriented access guarantees this representation.
      long entries = b.shape(1) * b.shape(2) * b.shape(3);
      nda::matrix_const_view<double> bf{{b.shape(0), 2 * entries}, reinterpret_cast<double const *>(b.data())};
      nda::matrix_view<double> cf{{c.shape(0), 2 * entries}, reinterpret_cast<double *>(c.data())};
      nda::blas::gemm(1., a, bf, accumulate, cf);
    }

    void real_vals2coefs(Input values, Output coefficients) const {
      coefficients = values;
      auto lu = itops.get_it2cf_lu();
      auto piv = itops.get_it2cf_piv();
      int r = itops.rank(), entries = static_cast<int>(2 * values.shape(1) * values.shape(2) * values.shape(3));
      double one = 1.;
      char side = 'R', upper = 'U', lower = 'L', transpose = 'N', nonunit = 'N', unit = 'U';
      auto *b = reinterpret_cast<double *>(coefficients.data());
      F77_dtrsm(&side, &upper, &transpose, &nonunit, &entries, &r, &one, lu.data(), &r, b, &entries);
      F77_dtrsm(&side, &lower, &transpose, &unit, &entries, &r, &one, lu.data(), &r, b, &entries);
      for (int k = r - 1; k >= 0; --k) {
        int other = piv(k) - 1;
        if (other != k) std::swap_ranges(b + long(k) * entries, b + long(k + 1) * entries, b + long(other) * entries);
      }
    }

    template <bool profile, typename F> static void stage(Stages &timing, int index, F &&f) {
      if constexpr (profile) {
        auto start = std::chrono::steady_clock::now();
        f();
        timing.seconds[index] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      } else f();
    }

    template <bool profile>
    void run(double beta, Input f, Input g, Output out, cppdlr::fast_convolve_batched_workspace &work, Policy policy, Stages &timing) const {
      check_array(f, itops.rank());
      check_array(g, itops.rank());
      check_array(out, itops.rank());
      if (!std::isfinite(beta) || beta <= 0. || f.shape(1) != g.shape(1) || out.shape(1) != f.shape(1)
          || f.shape(3) != g.shape(2) || out.shape(2) != f.shape(2) || out.shape(3) != g.shape(3))
        throw std::runtime_error("convolution experiment: incompatible input or beta");
      auto fits = [](auto const &storage, auto const &active) {
        return storage.shape(0) == active.shape(0) && storage.shape(1) >= active.shape(1) && storage.shape(2) == active.shape(2)
            && storage.shape(3) == active.shape(3);
      };
      if (!fits(work.fc, f) || !fits(work.hfc, f) || !fits(work.gc, g) || !fits(work.hgc, g) || !fits(work.product, out))
        throw std::runtime_error("convolution experiment: incompatible workspace");
      bool tiny = policy == Policy::fixed4 || policy == Policy::real_transform_fixed4 || policy == Policy::real_all_fixed4;
      bool real = policy != Policy::complex_blas && policy != Policy::fixed4;
      bool solve_real = policy == Policy::real_all || policy == Policy::real_all_fixed4;
      if (tiny && (f.shape(2) != 4 || f.shape(3) != 4 || g.shape(3) != 4))
        throw std::runtime_error("convolution experiment: fixed4 requires 4x4 matrices");
      Output fc{f.shape(), work.fc.data()}, gc{g.shape(), work.gc.data()}, hfc{f.shape(), work.hfc.data()}, hgc{g.shape(), work.hgc.data()};
      Output product{out.shape(), work.product.data()};
      stage<profile>(timing, 0, [&] {
        if (solve_real) {
          real_vals2coefs(f, fc);
          real_vals2coefs(g, gc);
        } else {
          production.vals2coefs(f, fc);
          production.vals2coefs(g, gc);
        }
      });
      auto apply_transform = [&](auto const &complex_matrix, auto const &real_matrix, Input input, Output output, double accumulate = 0.) {
        if (real) real_transform(real_matrix, input, output, accumulate);
        else transform(complex_matrix, input, output, accumulate);
      };
      stage<profile>(timing, 1, [&] {
        apply_transform(hilb, real_hilb, fc, hfc);
        apply_transform(hilb, real_hilb, gc, hgc);
      });
      constexpr auto all = nda::range::all;
      auto multiply = [&](auto a, auto b, auto c, double accumulate) {
        if (tiny) gemm4(1., a, b, accumulate, c);
        else nda::blas::gemm(1., a, b, accumulate, c);
      };
      stage<profile>(timing, 2, [&] {
        for (int l = 0; l < itops.rank(); ++l)
          for (long b = 0; b < f.shape(1); ++b) multiply(fc(l, b, all, all), gc(l, b, all, all), product(l, b, all, all), 0.);
      });
      stage<profile>(timing, 3, [&] { apply_transform(tcf2it, real_tcf2it, product, out); });
      stage<profile>(timing, 4, [&] {
        for (int l = 0; l < itops.rank(); ++l)
          for (long b = 0; b < f.shape(1); ++b) {
            multiply(hfc(l, b, all, all), gc(l, b, all, all), product(l, b, all, all), 0.);
            multiply(fc(l, b, all, all), hgc(l, b, all, all), product(l, b, all, all), 1.);
          }
      });
      stage<profile>(timing, 5, [&] { apply_transform(cf2it, real_cf2it, product, out, 1.); });
      stage<profile>(timing, 6, [&] { out *= beta; });
    }

    public:
    explicit Convolution(cppdlr::imtime_ops const &itops)
       : itops(itops), production(itops), cf2it(itops.get_cf2it()), hilb(itops.rank(), itops.rank()), tcf2it(hilb.shape()),
         real_cf2it(itops.get_cf2it()), real_hilb(hilb.shape()), real_tcf2it(hilb.shape()) {
      for (int j = 0; j < itops.rank(); ++j) {
        double t = itops.get_itnodes(j) > 0 ? itops.get_itnodes(j) : 1. + itops.get_itnodes(j);
        for (int k = 0; k < itops.rank(); ++k) {
          double k0 = cppdlr::k_it(0., itops.get_rfnodes(k));
          hilb(j, k) = j == k ? 0. : k0 / (itops.get_rfnodes(k) - itops.get_rfnodes(j));
          tcf2it(j, k) = t * cf2it(j, k) * k0;
          real_hilb(j, k) = hilb(j, k).real();
          real_tcf2it(j, k) = tcf2it(j, k).real();
        }
      }
    }
    Convolution(cppdlr::imtime_ops &&) = delete;

    void convolve(double beta, Input f, Input g, Output out, cppdlr::fast_convolve_batched_workspace &work, Policy policy) const {
      Stages unused;
      run<false>(beta, f, g, out, work, policy, unused);
    }
    void profile(double beta, Input f, Input g, Output out, cppdlr::fast_convolve_batched_workspace &work, Policy policy, Stages &timing) const {
      run<true>(beta, f, g, out, work, policy, timing);
    }
  };
} // namespace small_n

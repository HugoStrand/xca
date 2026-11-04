// Optional Eigen comparison; Eigen is not a dependency of the evaluator.
#include <triqs_xca/dense/small_gemm.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using Complex        = nda::dcomplex;
  using Stack          = nda::array<Complex, 3>;
  using Matrix4        = Eigen::Matrix<Complex, 4, 4, Eigen::RowMajor>;
  constexpr auto all   = nda::range::all;
  constexpr long rank  = 103;
  constexpr int trials = 5;
  enum class Kernel { handwritten, eigen, blas };
  constexpr std::array kernels{Kernel::handwritten, Kernel::eigen, Kernel::blas};
  constexpr std::array names{"handwritten", "eigen", "blas"};

  struct Case {
    char const *name;
    Complex alpha, beta;
  };

  // Constant alpha=1 matches evaluator products; general complex alpha is a separate control.
  template <Kernel K, bool UnitAlpha> void multiply_stack(Stack const &A, Stack const &B, Stack &C, Case const &test) {
    auto const alpha = UnitAlpha ? Complex{1.} : test.alpha;
    for (long t = 0; t < rank; ++t) {
      if constexpr (K == Kernel::handwritten) {
        triqs_xca::dense::small_gemm<4>(alpha, A(t, all, all), B(t, all, all), test.beta, C(t, all, all));
      } else if constexpr (K == Kernel::eigen) {
        Eigen::Map<Matrix4 const, Eigen::Unaligned> a(A.data() + 16 * t), b(B.data() + 16 * t);
        Eigen::Map<Matrix4, Eigen::Unaligned> c(C.data() + 16 * t);
        if (test.beta == Complex{})
          c.noalias() = alpha * a * b;
        else
          c.noalias() += alpha * a * b; // All accumulation cases have beta=1.
      } else {
        nda::blas::gemm(alpha, A(t, all, all), B(t, all, all), test.beta, C(t, all, all));
      }
    }
  }

  template <Kernel K> void checked_stack(Stack const &A, Stack const &B, Stack &C, Case const &test) {
    if (test.alpha == Complex{1.})
      multiply_stack<K, true>(A, B, C, test);
    else
      multiply_stack<K, false>(A, B, C, test);
  }

  void multiply_stack(Kernel kernel, Stack const &A, Stack const &B, Stack &C, Case const &test) {
    switch (kernel) {
      case Kernel::handwritten: checked_stack<Kernel::handwritten>(A, B, C, test); break;
      case Kernel::eigen: checked_stack<Kernel::eigen>(A, B, C, test); break;
      case Kernel::blas: checked_stack<Kernel::blas>(A, B, C, test); break;
    }
  }

  Stack reference(Stack const &A, Stack const &B, Stack const &initial, Case const &test) {
    using Wide = std::complex<long double>;
    auto wide  = [](Complex value) { return Wide{value.real(), value.imag()}; };
    Stack result(rank, 4, 4);
    for (long t = 0; t < rank; ++t)
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          Wide sum = 0.;
          for (int k = 0; k < 4; ++k) sum += wide(A(t, i, k)) * wide(B(t, k, j));
          auto value      = wide(test.alpha) * sum + wide(test.beta) * wide(initial(t, i, j));
          result(t, i, j) = {double(value.real()), double(value.imag())};
        }
    return result;
  }

  double check(Stack const &actual, Stack const &expected) {
    double error = 0., scale = 0.;
    for (long index = 0; index < actual.size(); ++index) {
      auto value = actual.data()[index];
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) throw std::runtime_error("non-finite GEMM result");
      error = std::max(error, std::abs(value - expected.data()[index]));
      scale = std::max(scale, std::abs(expected.data()[index]));
    }
    double relative = error / scale;
    if (!(relative < 5e-13)) throw std::runtime_error("GEMM differs from long-double reference");
    return relative;
  }

  Complex checksum(Stack const &values) {
    Complex sum = 0.;
    for (long index = 0; index < values.size(); ++index) sum += (1. + double(index % 17) / 17.) * values.data()[index];
    if (!std::isfinite(sum.real()) || !std::isfinite(sum.imag())) throw std::runtime_error("non-finite timing checksum");
    return sum;
  }

  template <Kernel K, bool UnitAlpha> double measure(Stack const &A, Stack const &B, Stack &C, Case const &test, long repeats) {
    auto start = std::chrono::steady_clock::now();
    for (long repeat = 0; repeat < repeats; ++repeat) {
      multiply_stack<K, UnitAlpha>(A, B, C, test);
      // Keep every overwritten stack observable to the optimizer, even for
      // beta=0. This is a compiler barrier, not a CPU fence or timed checksum.
      asm volatile("" : : "r"(C.data()) : "memory");
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return seconds * 1e9 / double(repeats * rank);
  }

  template <Kernel K> double measure_case(Stack const &A, Stack const &B, Stack &C, Case const &test, long repeats) {
    return test.alpha == Complex{1.} ? measure<K, true>(A, B, C, test, repeats) : measure<K, false>(A, B, C, test, repeats);
  }

  double measure(Kernel kernel, Stack const &A, Stack const &B, Stack &C, Case const &test, long repeats) {
    switch (kernel) {
      case Kernel::handwritten: return measure_case<Kernel::handwritten>(A, B, C, test, repeats);
      case Kernel::eigen: return measure_case<Kernel::eigen>(A, B, C, test, repeats);
      case Kernel::blas: return measure_case<Kernel::blas>(A, B, C, test, repeats);
    }
    throw std::logic_error("unknown GEMM kernel");
  }
} // namespace

int main(int argc, char **argv) {
  try {
    long repeats = argc > 1 ? std::stol(argv[1]) : 1024;
    if (argc > 2 || repeats < 1 || repeats > 1000000) throw std::invalid_argument("usage: eigen_gemm [repeats: 1..1000000]");
    Stack A(rank, 4, 4), B(rank, 4, 4), initial(rank, 4, 4);
    for (long index = 0; index < A.size(); ++index) {
      double x              = double(index + 1);
      A.data()[index]       = {std::sin(0.13 * x), 0.7 * std::cos(0.19 * x)};
      B.data()[index]       = {0.8 * std::cos(0.17 * x), std::sin(0.11 * x)};
      initial.data()[index] = {0.2 * std::cos(0.23 * x), 0.3 * std::sin(0.29 * x)};
    }
    std::array cases{Case{"overwrite_real_alpha", 1., 0.}, Case{"accumulate_real_alpha", 1., 1.}, Case{"overwrite_complex_alpha", {0.7, -0.3}, 0.},
                     Case{"accumulate_complex_alpha", {0.7, -0.3}, 1.}};
    std::cout << "case,kernel,N,rank,repeats,trials,median_ns,min_ns,max_ns,max_relative_error,checksum_real,checksum_imag\n";
    std::cout << std::setprecision(12);
    for (auto const &test : cases) {
      auto expected = reference(A, B, initial, test);
      std::array<double, 3> errors{};
      std::array<Complex, 3> checksums{};
      std::array<std::vector<double>, 3> timings;
      Stack output = initial;
      for (int k = 0; k < 3; ++k) {
        output = initial;
        multiply_stack(kernels[k], A, B, output, test); // Correctness and warmup.
        errors[k] = check(output, expected);
      }
      for (int trial = 0; trial < trials; ++trial)
        for (int offset = 0; offset < 3; ++offset) {
          int k  = (trial + offset) % 3;
          output = initial; // Initialization is outside the timed region.
          timings[k].push_back(measure(kernels[k], A, B, output, test, repeats));
          checksums[k] += checksum(output);
        }
      for (int k = 0; k < 3; ++k) {
        auto &times = timings[k];
        std::sort(times.begin(), times.end());
        std::cout << test.name << ',' << names[k] << ",4," << rank << ',' << repeats << ',' << trials << ',' << times[trials / 2] << ','
                  << times.front() << ',' << times.back() << ',' << errors[k] << ',' << checksums[k].real() << ',' << checksums[k].imag() << '\n';
      }
    }
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

#include <cppdlr/fast_convolve.hpp>
#include "legacy_convolve.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using Clock = std::chrono::steady_clock;

  struct Timing {
    double seconds;
    int repeats, trials;
  };

  template <typename F> Timing measure(F &&operation) {
    auto start = Clock::now();
    operation();
    double warm_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    // Target short batches to reduce timer noise without making slow cases long.
    int repeats = std::clamp(static_cast<int>(0.02 / std::max(warm_seconds, 1e-9)), 1, 1000);
    int trials  = warm_seconds > 0.1 ? 3 : 5;
    if (warm_seconds > 0.5) return {warm_seconds, 1, 1};
    std::vector<double> times;
    for (int trial = 0; trial < trials; ++trial) {
      start = Clock::now();
      for (int repeat = 0; repeat < repeats; ++repeat) operation();
      times.push_back(std::chrono::duration<double>(Clock::now() - start).count() / repeats);
    }
    std::ranges::sort(times);
    return {times[trials / 2], repeats, trials};
  }

  nda::array<nda::dcomplex, 3> values(cppdlr::imtime_ops const &itops, long dimension, int input) {
    nda::array<nda::dcomplex, 3> result(itops.rank(), dimension, dimension);
    std::array rates = input == 0 ? std::array{0.2, 1.3, 4.7} : std::array{0.2, 2.1, 3.6};
    double scale     = 1. / std::sqrt(dimension);
    for (long i = 0; i < dimension; ++i)
      for (long j = 0; j < dimension; ++j) {
        std::array<nda::dcomplex, 3> amplitudes;
        for (int k = 0; k < 3; ++k) {
          double phase  = 0.17 * (i + 1) * (j + 2) + 0.31 * (input + 1) * i + 0.23 * (k + 1) * j;
          amplitudes[k] = scale * nda::dcomplex{std::sin(phase + k), std::cos(phase + 0.37 * input + 0.29 * k)};
        }
        for (long l = 0; l < itops.rank(); ++l) {
          double t = itops.get_itnodes()(l);
          if (t < 0) t += 1.;
          result(l, i, j) = 0.;
          for (int k = 0; k < 3; ++k) result(l, i, j) += amplitudes[k] * std::exp(-rates[k] * t);
        }
      }
    return result;
  }

} // namespace

int main(int argc, char **argv) try {
  long largest_dimension = argc > 1 ? std::stol(argv[1]) : 512;
  if (argc > 2 || largest_dimension < 4 || largest_dimension > 1024)
    throw std::invalid_argument("usage: fast_convolve_benchmark [dimension upper bound: 4..1024, default 512]");

  constexpr double lambda = 1e5, eps = 1e-12, beta = 7.;
  constexpr double seconds_limit = 3., memory_limit = 2. * 1024 * 1024 * 1024;
  auto start = Clock::now();
  cppdlr::imtime_ops itops(lambda, cppdlr::build_dlr_rf(lambda, eps));
  double basis_seconds = std::chrono::duration<double>(Clock::now() - start).count();
  start                = Clock::now();
  cppdlr::fast_convolve convolution(itops);
  double kernel_seconds = std::chrono::duration<double>(Clock::now() - start).count();
  benchmark::legacy_convolve legacy_convolution(itops);
  std::cout << std::setprecision(10) << "# lambda=" << lambda << ", eps=" << eps << ", beta=" << beta << ", rank=" << itops.rank() << '\n'
            << "# basis_setup_ms=" << 1e3 * basis_seconds << ", fast_kernel_setup_ms=" << 1e3 * kernel_seconds << '\n';
#ifdef CPPDLR_GIT_HASH
#define XCA_BENCHMARK_STRING_IMPL(x) #x
#define XCA_BENCHMARK_STRING(x) XCA_BENCHMARK_STRING_IMPL(x)
  std::cout << "# cppdlr_git_hash=" << XCA_BENCHMARK_STRING(CPPDLR_GIT_HASH) << '\n';
#undef XCA_BENCHMARK_STRING
#undef XCA_BENCHMARK_STRING_IMPL
#endif
  std::cout << "# legacy_algorithm=cppdlr before b9b24ab; full Hilbert transforms inside the node loop\n";
  std::cout
     << "status,rank,dimension,reference_ms,fast_ms,speedup,relative_max_error,reference_repeats,reference_trials,fast_repeats,fast_trials,estimated_peak_mib,reason,"
        "legacy_status,legacy_ms,legacy_relative_max_error,legacy_repeats,legacy_trials,legacy_reason\n";

  double checksum = 0., last_slowest_seconds = 0., last_legacy_seconds = 0.;
  long last_dimension = 0, last_legacy_dimension = 0;
  for (long dimension = 4; dimension <= largest_dimension; dimension *= 4) {
    // Conservative peak bound includes both outputs, inputs, workspace and reference temporaries.
    double estimated_bytes   = 24. * sizeof(nda::dcomplex) * itops.rank() * dimension * dimension;
    double estimated_seconds = last_dimension ? last_slowest_seconds * std::pow(double(dimension) / last_dimension, 3) : 0.;
    if (dimension == 1024 && (estimated_seconds > seconds_limit || estimated_bytes > memory_limit)) {
      std::cout << "skipped," << itops.rank() << ',' << dimension << ",,,,,,,,," << estimated_bytes / (1024 * 1024) << ",projected "
                << estimated_seconds << " s per call; cap " << seconds_limit << " s or 2048 MiB,skipped,,,,,dimension skipped\n";
      continue;
    }

    auto f = values(itops, dimension, 0);
    auto g = values(itops, dimension, 1);
    nda::array<nda::dcomplex, 3> reference_out(itops.rank(), dimension, dimension), fast_out(reference_out.shape());
    cppdlr::fast_convolve_workspace workspace(itops.rank(), dimension, dimension, dimension);

    auto reference = [&] {
      auto fc       = itops.vals2coefs(f);
      auto gc       = itops.vals2coefs(g);
      reference_out = itops.convolve(beta, fc, gc, cppdlr::TIME_ORDERED);
      checksum += std::real(reference_out(itops.rank() / 2, dimension / 2, dimension / 2));
    };
    auto fast = [&] {
      convolution.convolve(beta, f, g, fast_out, workspace);
      checksum += std::real(fast_out(itops.rank() / 2, dimension / 2, dimension / 2));
    };
    // Allocated only when measuring the legacy case, after its runtime guard.
    nda::array<nda::dcomplex, 3> legacy_out;
    auto legacy = [&] {
      auto fc    = itops.vals2coefs(f);
      auto gc    = itops.vals2coefs(g);
      legacy_out = legacy_convolution.convolve(beta, fc, gc);
      checksum += std::real(legacy_out(itops.rank() / 2, dimension / 2, dimension / 2));
    };

    // Validate before timing; this also initializes the reference's cached convolution matrices.
    reference();
    fast();
    double relative_error = nda::max_element(nda::abs(fast_out - reference_out)) / std::max(1e-300, nda::max_element(nda::abs(reference_out)));
    if (!std::isfinite(relative_error) || relative_error > 2e-10) throw std::runtime_error("convolution outputs disagree");

    Timing ref_time      = measure(reference);
    Timing fast_time     = measure(fast);
    last_slowest_seconds = std::max(ref_time.seconds, fast_time.seconds);
    last_dimension       = dimension;

    std::optional<Timing> legacy_time;
    double legacy_error = 0.;
    std::ostringstream legacy_reason;
    double const projected_legacy_seconds = last_legacy_dimension ? last_legacy_seconds * std::pow(double(dimension) / last_legacy_dimension, 2) : 0.;
    if (dimension >= 256 && projected_legacy_seconds > seconds_limit) {
      // The repeated DLR transforms scale as D^2 at fixed r; spatial products add D^3 work.
      legacy_reason << "quadratic projection " << projected_legacy_seconds << " s per call; cap " << seconds_limit << " s";
    } else {
      legacy();
      legacy_error = nda::max_element(nda::abs(legacy_out - reference_out)) / std::max(1e-300, nda::max_element(nda::abs(reference_out)));
      if (!std::isfinite(legacy_error) || legacy_error > 2e-10) throw std::runtime_error("legacy convolution output disagrees");
      legacy_time           = measure(legacy);
      last_legacy_seconds   = legacy_time->seconds;
      last_legacy_dimension = dimension;
    }

    std::cout << "ok," << itops.rank() << ',' << dimension << ',' << 1e3 * ref_time.seconds << ',' << 1e3 * fast_time.seconds << ','
              << ref_time.seconds / fast_time.seconds << ',' << relative_error << ',' << ref_time.repeats << ',' << ref_time.trials << ','
              << fast_time.repeats << ',' << fast_time.trials << ',' << estimated_bytes / (1024 * 1024) << ",,";
    if (legacy_time)
      std::cout << "ok," << 1e3 * legacy_time->seconds << ',' << legacy_error << ',' << legacy_time->repeats << ',' << legacy_time->trials << ",\n";
    else
      std::cout << "skipped,,,,," << legacy_reason.str() << '\n';
    std::cout.flush();
  }
  std::cout << "# checksum=" << checksum << '\n';
  return std::isfinite(checksum) ? 0 : 1;
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

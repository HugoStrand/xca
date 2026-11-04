// Isolated research probe: cache the values-to-values matrix for one repeated left
// edge (G*K), then apply it to changing N=4 right operands. This is not a production
// evaluator. A batch deliberately shares the same left edge in every lane.
#include "convolution_experiment.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using C = nda::dcomplex;
using Clock = std::chrono::steady_clock;
constexpr long N = 4;
constexpr double beta = 2.;

struct Timing { double median, minimum, maximum; int repeats; };

// Different methods are interleaved between trials to reduce drift bias. Each
// method rotates among eight precomputed right operands during its timed loop.
std::vector<Timing> measure(std::vector<std::function<void()>> const &methods) {
  std::vector<int> repeats;
  for (auto const &method : methods) {
    method();
    auto start = Clock::now();
    for (int k = 0; k < 5; ++k) method();
    double seconds = std::chrono::duration<double>(Clock::now() - start).count() / 5.;
    repeats.push_back(std::clamp(int(0.01 / std::max(seconds, 1e-9)), 3, 2000));
  }
  std::vector<std::vector<double>> all(methods.size());
  for (int trial = 0; trial < 7; ++trial)
    for (int slot = 0; slot < int(methods.size()); ++slot) {
      int k = (slot + trial) % methods.size();
      auto start = Clock::now();
      for (int repeat = 0; repeat < repeats[k]; ++repeat) methods[k]();
      all[k].push_back(std::chrono::duration<double>(Clock::now() - start).count() / repeats[k]);
    }
  std::vector<Timing> result;
  for (int k = 0; k < int(methods.size()); ++k) {
    std::ranges::sort(all[k]);
    result.push_back({all[k][3], all[k].front(), all[k].back(), repeats[k]});
  }
  return result;
}

nda::array<C, 3> values(cppdlr::imtime_ops const &itops, int input) {
  nda::array<C, 3> v(itops.rank(), N, N);
  std::array rates = input == 0 ? std::array{0.2, 1.3, 4.7} : std::array{0.2, 2.1, 3.6};
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j)
      for (long l = 0; l < itops.rank(); ++l) {
        double t = itops.get_itnodes(l);
        if (t < 0.) t += 1.;
        v(l, i, j) = 0.;
        for (int k = 0; k < 3; ++k) {
          double phase = 0.17 * (i + 1) * (j + 2) + 0.031 * (input + 1) * i + 0.23 * (k + 1) * j;
          C amplitude{std::sin(phase + k), std::cos(phase + 0.037 * input + 0.29 * k)};
          v(l, i, j) += 0.5 * amplitude * std::exp(-(rates[k] + (input == 0 ? 0.4 : 0.)) * t);
        }
      }
  return v;
}

struct CachedConvolution {
  nda::matrix<C> matrix, packed, product;
  long rank, batch;
  CachedConvolution(nda::matrix<C> matrix, long rank, long batch)
     : matrix(std::move(matrix)), packed(rank * N, batch * N), product(packed.shape()), rank(rank), batch(batch) {}

  void apply(nda::array_const_view<C, 4> input, nda::array_view<C, 4> output) {
    if (batch == 1) {
      auto in = nda::reshape(input, rank * N, N);
      auto out = nda::reshape(output, rank * N, N);
      if (input.data() == output.data()) {
        // GEMM does not allow overlap, unlike fast_convolve. A dead evaluator
        // buffer could hold this output; this probe includes the fallback copy.
        nda::blas::gemm(1., matrix, in, 0., product);
        out = product;
      } else {
        nda::blas::gemm(1., matrix, in, 0., out);
      }
      return;
    }
    // Current evaluator layout is (tau,batch,row,col). One shared left matrix
    // needs (tau,row,batch,col) so all batch right-hand sides are columns.
    for (long t = 0; t < rank; ++t)
      for (long b = 0; b < batch; ++b)
        for (long i = 0; i < N; ++i)
          for (long j = 0; j < N; ++j) packed(t * N + i, b * N + j) = input(t, b, i, j);
    nda::blas::gemm(1., matrix, packed, 0., product);
    for (long t = 0; t < rank; ++t)
      for (long b = 0; b < batch; ++b)
        for (long i = 0; i < N; ++i)
          for (long j = 0; j < N; ++j) output(t, b, i, j) = product(t * N + i, b * N + j);
  }
};

double relative_error(auto const &value, auto const &reference) {
  for (auto const *array : {&value, &reference})
    for (long k = 0; k < array->size(); ++k)
      if (!std::isfinite(array->data()[k].real()) || !std::isfinite(array->data()[k].imag()))
        throw std::runtime_error("nonfinite convolution output");
  return nda::max_element(nda::abs(value - reference)) / std::max(1e-300, nda::max_element(nda::abs(reference)));
}

void run(double lambda, double eps, long batch, bool check_only, double &checksum) {
  cppdlr::imtime_ops itops(lambda, cppdlr::build_dlr_rf(lambda, eps));
  cppdlr::fast_convolve fast(itops);
  small_n::Convolution improved(itops);
  constexpr auto improved_policy = small_n::Policy::real_all_fixed4;
  long r = itops.rank();
  auto edge = values(itops, 0);
  auto start = Clock::now();
  auto edge_coefficients = itops.vals2coefs(edge);
  auto matrix = itops.convmat(beta, edge_coefficients, true);
  double cold_setup = std::chrono::duration<double>(Clock::now() - start).count();
  CachedConvolution cached(std::move(matrix), r, batch);
  cppdlr::fast_convolve_batched_workspace workspace(r, batch, N, N, N);
  nda::array<C, 4> left(r, batch, N, N), fast_out(left.shape()), cached_out(left.shape()), alias_out(left.shape()), reference(left.shape()), improved_out(left.shape());
  std::vector<nda::array<C, 4>> inputs;
  for (int sample = 0; sample < 8; ++sample) {
    inputs.emplace_back(left.shape());
    for (long b = 0; b < batch; ++b) {
      auto v = values(itops, 1 + sample * batch + b);
      for (long t = 0; t < r; ++t)
        for (long i = 0; i < N; ++i)
          for (long j = 0; j < N; ++j) {
            inputs.back()(t,b,i,j) = v(t,i,j);
            left(t,b,i,j) = edge(t,i,j);
          }
    }
  }
  double error_fast = 0., error_direct = 0., error_alias = 0., fast_direct = 0., improved_error = 0., improved_alias_error = 0.;
  for (auto const &input : inputs) {
    fast.convolve(beta, left, input, fast_out, workspace);
    cached.apply(input, cached_out);
    alias_out = input;
    cached.apply(alias_out, alias_out);
    for (long b = 0; b < batch; ++b) {
      nda::array<C, 3> g(r,N,N);
      for (long t = 0; t < r; ++t)
        for (long i = 0; i < N; ++i)
          for (long j = 0; j < N; ++j) g(t,i,j) = input(t,b,i,j);
      auto direct = itops.convolve(beta, edge_coefficients, itops.vals2coefs(g), true);
      for (long t = 0; t < r; ++t)
        for (long i = 0; i < N; ++i)
          for (long j = 0; j < N; ++j) reference(t,b,i,j) = direct(t,i,j);
    }
    error_fast = std::max(error_fast, relative_error(cached_out, fast_out));
    error_direct = std::max(error_direct, relative_error(cached_out, reference));
    error_alias = std::max(error_alias, relative_error(alias_out, fast_out));
    fast_direct = std::max(fast_direct, relative_error(fast_out, reference));
    improved.convolve(beta, left, input, improved_out, workspace, improved_policy);
    improved_error = std::max(improved_error, relative_error(improved_out, fast_out));
    alias_out = input;
    improved.convolve(beta, left, alias_out, alias_out, workspace, improved_policy);
    improved_alias_error = std::max(improved_alias_error, relative_error(alias_out, fast_out));
  }
  std::cerr << "rank=" << r << " batch=" << batch << " convmat_vs_fast=" << error_fast
            << " convmat_vs_direct=" << error_direct << " alias=" << error_alias << " fast_vs_direct=" << fast_direct
            << " improved_vs_fast=" << improved_error << " improved_alias_vs_fast=" << improved_alias_error << '\n';
  for (double error : {error_fast, error_direct, error_alias, fast_direct, improved_error, improved_alias_error})
    if (!std::isfinite(error) || error > 2e-10) throw std::runtime_error("convolution outputs disagree");
  if (check_only) return;
  int fcounter=0, ccounter=0, acounter=0, facounter=0, icounter=0, iacounter=0;
  // Both fast and cached alias rows include the same source copy. The noalias
  // rows are the useful kernel comparison; an evaluator can swap output owners.
  std::vector<std::function<void()>> methods;
  methods.emplace_back([&] {
    auto const &input = inputs[(fcounter++) % inputs.size()];
    fast.convolve(beta, left, input, fast_out, workspace);
    checksum += fast_out(r/2,0,1,2).real();
  });
  methods.emplace_back([&] {
    auto const &input = inputs[(ccounter++) % inputs.size()];
    cached.apply(input, cached_out);
    checksum += cached_out(r/2,0,1,2).real();
  });
  methods.emplace_back([&] {
    alias_out = inputs[(acounter++) % inputs.size()];
    cached.apply(alias_out, alias_out);
    checksum += alias_out(r/2,0,1,2).real();
  });
  methods.emplace_back([&] {
    alias_out = inputs[(facounter++) % inputs.size()];
    fast.convolve(beta, left, alias_out, alias_out, workspace);
    checksum += alias_out(r/2,0,1,2).real();
  });
  methods.emplace_back([&] {
    auto const &input = inputs[(icounter++) % inputs.size()];
    improved.convolve(beta, left, input, improved_out, workspace, improved_policy);
    checksum += improved_out(r/2,0,1,2).real();
  });
  methods.emplace_back([&] {
    alias_out = inputs[(iacounter++) % inputs.size()];
    improved.convolve(beta, left, alias_out, alias_out, workspace, improved_policy);
    checksum += alias_out(r/2,0,1,2).real();
  });
  auto measurements = measure(methods);
  auto setup = measure({[&] {
    auto fc = itops.vals2coefs(edge);
    auto fresh = itops.convmat(beta, fc, true);
    checksum += fresh(r*N/2,r*N/2).real();
  }}).front();
  std::array names{"fast_batched", "cached_convmat", "cached_alias_with_reset", "fast_alias_with_reset",
                   "real_all_fixed4", "real_all_fixed4_alias_with_reset"};
  for (int k=0;k<6;++k) {
    auto time = measurements[k];
    bool alias = k == 2 || k == 3 || k == 5;
    double baseline = measurements[alias ? 3 : 0].median;
    double improved_baseline = measurements[alias ? 5 : 4].median;
    auto amortization = [&](double reference) {
      // Setup is a cost of cached variants only. For a non-improving cached
      // application no finite reuse count can recover construction cost.
      if (k != 1 && k != 2) return std::numeric_limits<double>::quiet_NaN();
      double saved = reference - time.median;
      return saved > 0. ? setup.median / saved : std::numeric_limits<double>::infinity();
    };
    std::cout << r << ',' << N << ',' << batch << ',' << names[k] << ',' << 1e6*time.median << ',' << 1e6*time.minimum << ',' << 1e6*time.maximum
              << ',' << time.repeats << ',' << baseline/time.median << ',' << 1e6*setup.median << ',' << 1e6*cold_setup
              << ',' << amortization(baseline) << ',' << r*N*r*N*sizeof(C) << ',' << error_fast << ',' << error_direct << ',' << error_alias << ',' << fast_direct
              << ',' << improved_baseline/time.median << ',' << amortization(improved_baseline) << ',' << (alias ? "alias_with_reset" : "noalias")
              << ',' << improved_error << ',' << improved_alias_error << '\n';
  }
}
} // namespace

int main(int argc, char **argv) try {
  bool check_only = argc > 1 && std::string(argv[1]) == "--check";
  if (argc > (check_only ? 2 : 1)) throw std::invalid_argument("usage: convmat_probe [--check]");
  std::cout << std::setprecision(10)
            << "# fixed shared left edge, changing smooth complex noncommuting RHS; beta=2; 7 rotated trials\n"
            << "# cached_convmat includes packing and unpacking at batch>1; alias_with_reset includes input restoration for all implementations\n"
            << "# speedups and amortization use matching alias/noalias baselines; amortization is nan on noncached rows\n"
            << "rank,N,batch,variant,median_us,min_us,max_us,repeats,speedup_vs_fast,build_us,cold_build_us,amortization_calls,matrix_bytes,convmat_vs_fast,convmat_vs_direct,alias_vs_fast,fast_vs_direct,speedup_vs_improved,amortization_calls_vs_improved,comparison_mode,improved_vs_fast,improved_alias_vs_fast\n";
  double checksum=0.;
  for (auto [lambda,eps] : {std::pair{20.,1e-6}, std::pair{1e5,1e-12}})
    for (long batch : {1,8}) run(lambda,eps,batch,check_only,checksum);
  std::cerr << "checksum=" << checksum << '\n';
} catch (std::exception const &error) {
  std::cerr << error.what() << '\n';
  return 1;
}

#include "convolution_experiment.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
  using Complex = nda::dcomplex;
  using Tensor = nda::array<Complex, 4>;
  using Clock = std::chrono::steady_clock;
  constexpr double beta = 7.;
  constexpr std::array rates_f{0.2, 1.3, 4.7}, rates_g{0.2, 2.1, 3.6};
  constexpr auto all = nda::range::all;
  double checksum = 0.;

  Complex amplitude(int input, long lane, int pole, int i, int j) {
    double phase = 0.17 * (i + 1) * (j + 2) + 0.31 * (input + 1) * i + 0.23 * (pole + 1) * j + 0.19 * lane;
    return 0.5 * Complex{std::sin(phase + pole), std::cos(phase + 0.37 * input + 0.29 * pole)};
  }

  Tensor values(cppdlr::imtime_ops const &itops, long batch, int input) {
    Tensor result(itops.rank(), batch, 4, 4);
    auto const &rates = input == 0 ? rates_f : rates_g;
    for (int l = 0; l < itops.rank(); ++l) {
      double t = itops.get_itnodes(l);
      if (t < 0) t += 1.;
      for (long b = 0; b < batch; ++b)
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            result(l, b, i, j) = 0.;
            for (int pole = 0; pole < 3; ++pole) result(l, b, i, j) += amplitude(input, b, pole, i, j) * std::exp(-rates[pole] * t);
          }
    }
    return result;
  }

  Tensor analytic(cppdlr::imtime_ops const &itops, long batch) {
    Tensor result(itops.rank(), batch, 4, 4);
    for (int l = 0; l < itops.rank(); ++l) {
      double t = itops.get_itnodes(l);
      if (t < 0) t += 1.;
      for (long b = 0; b < batch; ++b)
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            result(l, b, i, j) = 0.;
            for (int p = 0; p < 3; ++p)
              for (int q = 0; q < 3; ++q) {
                double difference = rates_g[q] - rates_f[p];
                double integral = difference == 0 ? t * std::exp(-rates_f[p] * t)
                                                  : std::exp(-rates_f[p] * t) * (-std::expm1(-difference * t)) / difference;
                for (int k = 0; k < 4; ++k)
                  result(l, b, i, j) += beta * integral * amplitude(0, b, p, i, k) * amplitude(1, b, q, k, j);
              }
          }
    }
    return result;
  }

  double error(Tensor const &actual, Tensor const &reference) {
    if (actual.shape() != reference.shape()) throw std::runtime_error("result shape differs");
    double maximum = 0., norm = 0.;
    for (long i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(actual.data()[i].real()) || !std::isfinite(actual.data()[i].imag())
          || !std::isfinite(reference.data()[i].real()) || !std::isfinite(reference.data()[i].imag()))
        throw std::runtime_error("nonfinite convolution result");
      norm = std::max(norm, std::abs(reference.data()[i]));
      maximum = std::max(maximum, std::abs(actual.data()[i] - reference.data()[i]));
    }
    if (norm < 1e-10) throw std::runtime_error("uninformative zero reference");
    return maximum / norm;
  }

  void check_gemm4() {
    nda::matrix<Complex> a(8, 8), b(8, 8), c(8, 8);
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        a(i, j) = amplitude(0, 2, 1, i, j);
        b(i, j) = amplitude(1, 1, 2, i, j);
        c(i, j) = amplitude(1, 3, 0, i, j);
      }
    auto av = a(nda::range(0, 8, 2), nda::range(0, 8, 2));
    auto bv = b(nda::range(0, 8, 2), nda::range(0, 8, 2));
    auto cv = c(nda::range(0, 8, 2), nda::range(0, 8, 2));
    for (Complex alpha : {Complex{1., 0.}, Complex{0.3, -0.7}})
      for (Complex beta : {Complex{}, Complex{1., 0.}, Complex{-0.1, 0.2}}) {
        nda::matrix<Complex> reference(cv);
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            Complex sum{};
            for (int k = 0; k < 4; ++k) sum += av(i, k) * bv(k, j);
            reference(i, j) = alpha * sum + beta * reference(i, j);
          }
        small_n::gemm4(alpha, av, bv, beta, cv);
        if (nda::max_element(nda::abs(cv - reference)) > 3e-14) throw std::runtime_error("gemm4 strided alpha/beta check failed");
      }
  }

  struct Variant {
    char const *name;
    std::function<void()> operation;
    small_n::Policy policy;
    int repeats = 0;
    double production_error = 0., analytic_error = 0.;
    std::vector<double> timings;
  };

  void run_case(double lambda, double eps, long expected_rank, long batch, bool check_only) {
    cppdlr::imtime_ops itops(lambda, cppdlr::build_dlr_rf(lambda, eps));
    if (itops.rank() != expected_rank) throw std::runtime_error("unexpected DLR rank");
    cppdlr::fast_convolve production(itops);
    small_n::Convolution experiment(itops);
    // Deliberately use capacity 32 at every active batch, checking packed tails.
    cppdlr::fast_convolve_batched_workspace workspace(itops.rank(), 32, 4, 4, 4);
    auto f = values(itops, batch, 0), g = values(itops, batch, 1), exact = analytic(itops, batch);
    Tensor result(f.shape()), reference(f.shape());
    production.convolve(beta, f, g, reference, workspace);
    std::vector<Variant> variants;
    variants.push_back({"production", [&] { production.convolve(beta, f, g, result, workspace); }, small_n::Policy::complex_blas});
    for (auto policy : small_n::policies)
      variants.push_back({small_n::name(policy), [&, policy] { experiment.convolve(beta, f, g, result, workspace, policy); }, policy});
    for (auto &v : variants) {
      v.operation();
      v.production_error = error(result, reference);
      v.analytic_error = error(result, exact);
      if (v.production_error > 2e-10 || v.analytic_error > std::max(20 * eps, 2e-10))
        throw std::runtime_error(std::string(v.name) + ": convolution disagrees with production or analytic values");
      if (std::string(v.name) == "complex_blas" && v.production_error != 0.)
        throw std::runtime_error("complex_blas clone does not exactly match production");
      if (std::string(v.name) != "production") {
        // Output overwriting either input must retain exactly the intended semantics.
        Tensor alias(f);
        experiment.convolve(beta, alias, g, alias, workspace, v.policy);
        if (error(alias, result) > 1e-14) throw std::runtime_error("f alias check failed");
        alias = g;
        experiment.convolve(beta, f, alias, alias, workspace, v.policy);
        if (error(alias, result) > 1e-14) throw std::runtime_error("g alias check failed");
      }
      if (check_only) continue;
      auto start = Clock::now();
      v.operation();
      double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      v.repeats = std::clamp(int(0.005 / std::max(seconds, 1e-9)), 2, 2000);
    }
    constexpr int trials = 5;
    if (!check_only)
      for (int trial = 0; trial < trials; ++trial)
        for (int offset = 0; offset < int(variants.size()); ++offset) {
          auto &v = variants[(trial + offset) % variants.size()];
          auto start = Clock::now();
          for (int repeat = 0; repeat < v.repeats; ++repeat) {
            v.operation();
            checksum += result(itops.rank() / 2, batch / 2, 0, 0).real();
          }
          v.timings.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / v.repeats);
        }
    for (auto &v : variants) {
      std::cout << itops.rank() << ',' << batch << ',' << v.name << ',';
      if (check_only) std::cout << ",,,,0,0,";
      else {
        std::ranges::sort(v.timings);
        double median = v.timings[trials / 2];
        std::cout << median << ',' << v.timings.front() << ',' << v.timings.back() << ',' << median / batch << ',' << v.repeats << ',' << trials << ',';
      }
      std::cout << v.production_error << ',' << v.analytic_error;
      if (!check_only && std::string(v.name) != "production") {
        small_n::Stages timing;
        // Aggregate stage timings are a separate instrumented pass, with each stage
        // clocked. Never derive the uninstrumented total from their sum.
        int repeats = std::max(5, v.repeats);
        for (int i = 0; i < repeats; ++i) experiment.profile(beta, f, g, result, workspace, v.policy, timing);
        for (double seconds : timing.seconds) std::cout << ',' << seconds * 1e6 / repeats;
        std::cout << ',' << repeats;
      } else std::cout << ",,,,,,,,0";
      std::cout << '\n';
    }
    std::cout.flush();
  }
} // namespace

int main(int argc, char **argv) try {
  bool check_only = argc == 2 && std::string(argv[1]) == "--check-only";
  if (argc > 1 && !check_only) throw std::runtime_error("usage: convolution_probe [--check-only]");
  check_gemm4();
  std::cout << std::setprecision(12)
            << "# N=4; beta=7; smooth three-pole complex noncommuting matrices; no fast-math\n"
            << "# profile columns come from a separate instrumented pass, not the uninstrumented median\n"
            << "rank,batch,variant,median_us,min_us,max_us,per_lane_us,repeats,trials,error_vs_production,error_vs_analytic";
  for (auto const *name : small_n::Stages::names) std::cout << ',' << name << "_us";
  std::cout << ",profile_repeats\n";
  for (long batch : {1, 8, 32}) run_case(20., 1e-6, 13, batch, check_only);
  for (long batch : {1, 8, 32}) run_case(1e5, 1e-12, 103, batch, check_only);
  std::cout << "# checksum=" << checksum << '\n';
  return std::isfinite(checksum) ? 0 : 1;
} catch (std::exception const &e) {
  std::cerr << e.what() << '\n';
  return 1;
}

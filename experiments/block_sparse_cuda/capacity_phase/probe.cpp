// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
namespace gpu = triqs_xca::block_sparse::cuda;
using Z       = nda::dcomplex;
using LZ      = std::complex<long double>;
using Clock   = std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
Z coefficient(long t, long i, long j, long r, long m, long n, double phase) {
  return Z{std::sin(.7 * t + .3 * i - .2 * j + phase), std::cos(.1 * t - .4 * i + .8 * j + phase)} / double(r * std::max(m, n));
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
std::vector<LZ> oracle(cppdlr::imtime_ops const &host, long M, long K, long N, long i, long j) {
  long r  = host.rank();
  auto rf = host.get_rfnodes(), it = host.get_itnodes();
  auto C = host.get_cf2it();
  std::vector<LZ> f(r * K), g(r * K), prod(r), cross(r), result(r);
  for (long t = 0; t < r; ++t)
    for (long k = 0; k < K; ++k) {
      f[t * K + k] = LZ(coefficient(t, i, k, r, M, K, .3));
      g[t * K + k] = LZ(coefficient(t, k, j, r, K, N, .8));
    }
  for (long t = 0; t < r; ++t)
    for (long k = 0; k < K; ++k) {
      LZ hf{}, hg{};
      for (long s = 0; s < r; ++s) {
        // Independently assemble the same cached double constants, then use
        // long-double sums/products without sharing device arithmetic.
        double h = t == s ? 0. : cppdlr::k_it(0., rf(s)) / (rf(s) - rf(t));
        hf += static_cast<long double>(h) * f[s * K + k];
        hg += static_cast<long double>(h) * g[s * K + k];
      }
      prod[t] += f[t * K + k] * g[t * K + k];
      cross[t] += hf * g[t * K + k] + f[t * K + k] * hg;
    }
  for (long t = 0; t < r; ++t)
    for (long s = 0; s < r; ++s) {
      double q = (it(t) > 0 ? it(t) : 1. + it(t)) * C(t, s) * cppdlr::k_it(0., rf(s));
      result[t] += 2.L * (static_cast<long double>(q) * prod[s] + static_cast<long double>(C(t, s)) * cross[s]);
    }
  return result;
}
int main(int argc, char **argv) {
  try {
    long M = 65, K = 64, N = 63, samples = 1, budget_mib = 1024;
    double lambda = 20.;
    bool dry = false, large = false;
    for (int a = 1; a < argc; ++a) {
      std::string key = argv[a];
      if (key == "--dry-run") {
        dry = true;
        continue;
      }
      if (key == "--allow-large") {
        large = true;
        continue;
      }
      if (++a == argc) throw std::invalid_argument("Missing option value");
      std::size_t used  = 0;
      std::string value = argv[a];
      if (key == "--lambda") {
        lambda = std::stod(value, &used);
      } else {
        long v = std::stol(value, &used);
        if (key == "--m")
          M = v;
        else if (key == "--k")
          K = v;
        else if (key == "--n")
          N = v;
        else if (key == "--samples")
          samples = v;
        else if (key == "--budget-mib")
          budget_mib = v;
        else
          throw std::invalid_argument("Unknown option");
      }
      if (used != value.size()) throw std::invalid_argument("Invalid option suffix");
    }
    if (M < 1 || K < 1 || N < 1 || M > 3432 || K > 3432 || N > 3432 || samples < 1 || samples > 2 || budget_mib < 1 || budget_mib > 98304
        || !std::isfinite(lambda) || lambda < 10. || lambda > 10000.)
      throw std::invalid_argument("Beyond bounded capacity probe");
    auto begin = Clock::now();
    cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, 1e-10, false), false);
    long r = host.rank(), D = std::max({M, K, N});
    auto tables                   = gpu::checked_add(gpu::checked_multiply(5, gpu::tensor_bytes(1, r, r)), gpu::checked_multiply(r + 1, sizeof(int)));
    auto scratch                  = gpu::checked_add(gpu::checked_add(gpu::tensor_bytes(r, M, K), gpu::tensor_bytes(r, K, N)),
                                                     gpu::checked_multiply(2, gpu::tensor_bytes(r, M, N)));
    constexpr std::size_t reserve = 32ULL * 1024 * 1024, headroom = 64ULL * 1024 * 1024;
    auto needed = gpu::checked_add(
       tables,
       gpu::checked_add(
          reserve,
          gpu::checked_add(
             gpu::tensor_bytes(r, D, D),
             gpu::checked_add(
                scratch, gpu::checked_add(gpu::tensor_bytes(r, M, K), gpu::checked_add(gpu::tensor_bytes(r, K, N), gpu::tensor_bytes(r, M, N)))))));
    for (auto bytes : {gpu::tensor_bytes(r, M, K), gpu::tensor_bytes(r, K, N), gpu::tensor_bytes(r, M, N)})
      if (bytes / sizeof(Z) > std::size_t(std::numeric_limits<int>::max())) throw std::overflow_error("Phase tensor exceeds BLAS vector range");
    auto budget = gpu::checked_multiply(budget_mib, 1024ULL * 1024);
    std::cerr << "lambda=" << lambda << " rank=" << r << " explicit_bytes=" << needed << " budget_bytes=" << budget << " dry_run=" << dry << '\n';
    if (dry) return 0; // Host-only inventory: no CUDA context or large payload allocation.
    if (needed > budget) {
      std::cerr << "Explicit phase budget rejected before device allocation\n";
      return 2;
    }
    if (D > 512) {
      cudaDeviceProp prop{};
      int device = -1;
      check(cudaGetDevice(&device));
      check(cudaGetDeviceProperties(&prop, device));
      if (!large || !std::getenv("SLURM_JOB_ID") || std::string(prop.name).find("GH200") == std::string::npos)
        throw std::invalid_argument("Large execution requires --allow-large inside a GH200 Slurm job");
    }
    auto ops = std::make_shared<gpu::DeviceOps>();
    gpu::DlrOps dlr(ops, host, gpu::checked_add(tables, scratch));
    std::size_t baseline = 0, total = 0;
    check(cudaMemGetInfo(&baseline, &total));
    if (gpu::checked_add(needed - tables, headroom) > baseline) {
      std::cerr << "Free-memory/headroom phase admission rejected\n";
      return 2;
    }
    gpu::Tensor f, g, out, opposite;
    nda::cuvector<unsigned char> operator_reserve;
    try {
      f                = gpu::Tensor(r, M, K);
      g                = gpu::Tensor(r, K, N);
      out              = gpu::Tensor(r, M, N);
      opposite         = gpu::Tensor(r, D, D);
      operator_reserve = nda::cuvector<unsigned char>(reserve);
      check(cudaMemset(operator_reserve.data(), 0, reserve));
      ops->zero(opposite);
      auto upload = [&](gpu::Tensor &target, double phase) {
        nda::matrix<Z> plane(target.extent(1), target.extent(2));
        for (long t = 0; t < r; ++t) {
          for (long i = 0; i < plane.extent(0); ++i)
            for (long j = 0; j < plane.extent(1); ++j) plane(i, j) = coefficient(t, i, j, r, plane.extent(0), plane.extent(1), phase);
          check(cudaMemcpy(target.data() + t * plane.size(), plane.data(), plane.size() * sizeof(Z), cudaMemcpyHostToDevice));
        }
      };
      upload(f, .3);
      upload(g, .8);
      ops->wait_for_default_uploads();
      ops->finish();
      double setup_ms   = ms(begin);
      auto oracle_start = Clock::now();
      std::vector<std::pair<long, long>> points{{0, 0}, {M / 2, N / 2}, {M - 1, N - 1}};
      std::vector<std::vector<LZ>> gold;
      for (auto [i, j] : points) gold.push_back(oracle(host, M, K, N, i, j));
      double oracle_ms = ms(oracle_start);
      std::vector<Z> previous;
      std::cout << std::setprecision(17)
                << "rank,m,k,n,sample,setup_ms,oracle_ms,convolve_ms,explicit_bytes,sampled_delta,max_abs_error,oracle_norm,retained_dlr_bytes\n";
      for (long sample = 0; sample < samples; ++sample) {
        auto start = Clock::now();
        dlr.convolve(2., f, g, out);
        ops->finish();
        double elapsed   = ms(start);
        std::size_t free = 0;
        check(cudaMemGetInfo(&free, &total));
        double error = 0., norm = 0.;
        std::vector<Z> actual;
        for (std::size_t p = 0; p < points.size(); ++p)
          for (long t = 0; t < r; ++t) {
            auto [i, j] = points[p];
            Z z;
            check(cudaMemcpy(&z, out.data() + (t * M + i) * N + j, sizeof(Z), cudaMemcpyDeviceToHost));
            if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::runtime_error("Nonfinite sampled output");
            error = std::max(error, double(std::abs(LZ(z) - gold[p][t])));
            norm  = std::max(norm, double(std::abs(gold[p][t])));
            actual.push_back(z);
          }
        if (error > 1e-11 * norm + 1e-24) throw std::runtime_error("Independent sampled oracle failed");
        if (sample && actual != previous) throw std::runtime_error("Sampled replay was not exact");
        previous = std::move(actual);
        std::cout << r << ',' << M << ',' << K << ',' << N << ',' << sample << ',' << setup_ms << ',' << oracle_ms << ',' << elapsed << ',' << needed
                  << ',' << (free < baseline ? baseline - free : 0) << ',' << error << ',' << norm << ',' << dlr.storage_bytes() << '\n';
      }
      dlr.release_scratch();
      if (dlr.storage_bytes() != tables) throw std::runtime_error("Scratch release did not retain only tables");
    } catch (...) {
      try {
        ops->finish();
      } catch (...) {}
      throw;
    }
    return 0;
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

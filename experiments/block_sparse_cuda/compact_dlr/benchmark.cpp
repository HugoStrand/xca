// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "compact_dlr.hpp"
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  using Array   = nda::array<Complex, 3>;
  Array coefficients(long r, long m, long n, double phase) {
    Array out(r, m, n);
    for (long t = 0; t < r; ++t)
      for (long i = 0; i < m; ++i)
        for (long j = 0; j < n; ++j)
          out(t, i, j) = Complex{std::sin(.7 * t + .3 * i - .2 * j + phase), std::cos(.1 * t - .4 * i + .8 * j + phase)} / (r * std::max(m, n));
    return out;
  }
} // namespace
int main(int argc, char **argv) {
  try {
    int M = 4, K = 6, N = 3, samples = 3;
    double lambda = 20.;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (++i == argc) throw std::invalid_argument("Missing value");
      if (key == "--m")
        M = std::stoi(argv[i]);
      else if (key == "--k")
        K = std::stoi(argv[i]);
      else if (key == "--n")
        N = std::stoi(argv[i]);
      else if (key == "--samples")
        samples = std::stoi(argv[i]);
      else if (key == "--lambda")
        lambda = std::stod(argv[i]);
      else
        throw std::invalid_argument("Unknown option");
    }
    if (M < 1 || K < 1 || N < 1 || M > 256 || K > 256 || N > 256 || samples < 1 || samples > 5 || !std::isfinite(lambda) || lambda < 10
        || lambda > 120)
      throw std::invalid_argument("Beyond bounded DLR replay");
    cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, 1e-10, false), false);
    long r     = host.rank();
    auto owner = std::make_shared<gpu::DeviceOps>();
    gpu::DlrOps cached(owner, host, 1024ULL * 1024 * 1024);
    gpu::CompactDlrOps compact(owner, host, 1024ULL * 1024 * 1024);
    auto fc = coefficients(r, M, K, .3), gc = coefficients(r, K, N, .8);
    gpu::Tensor fv(host.coefs2vals(fc)), gv(host.coefs2vals(gc)), fd(fc.shape()), gd(gc.shape()), out(r, M, N);
    auto run = [&](auto &device) {
      std::size_t peak = device.storage_bytes();
      device.vals2coefs(fv, fd);
      peak = std::max(peak, device.storage_bytes());
      device.vals2coefs(gv, gd);
      peak = std::max(peak, device.storage_bytes());
      device.convolve(2., fd, gd, out);
      peak = std::max(peak, device.storage_bytes());
      return peak;
    };
    run(cached);
    auto gold = nda::to_host(out);
    run(compact);
    auto check = [&] {
      auto actual  = nda::to_host(out);
      double error = 0.;
      for (long i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual.data()[i].real()) || !std::isfinite(actual.data()[i].imag())) throw std::runtime_error("Nonfinite replay");
        error = std::max(error, std::abs(actual.data()[i] - gold.data()[i]));
      }
      if (error != 0.) throw std::runtime_error("Compact/cached replay differs");
    };
    check();
    std::cout << std::setprecision(17) << "route,rank,m,k,n,sample,call_ms,dlr_retained_bytes,dlr_observed_boundary_peak_bytes\n";
    for (int sample = 0; sample < samples; ++sample)
      for (bool low : {false, true}) {
        auto start     = std::chrono::steady_clock::now();
        auto peak      = low ? run(compact) : run(cached);
        double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        check();
        std::cout << (low ? "compact" : "cached") << ',' << r << ',' << M << ',' << K << ',' << N << ',' << sample << ',' << elapsed << ','
                  << (low ? compact.storage_bytes() : cached.storage_bytes()) << ',' << peak << '\n';
      }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

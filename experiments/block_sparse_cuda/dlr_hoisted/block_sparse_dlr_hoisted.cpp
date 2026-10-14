// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "hoisted.hpp"
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {
  using block_sparse_dlr_experiment::Array;
  using Clock = std::chrono::steady_clock;
  double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
  Array coefficients(long rank, long m, long n, double phase) {
    Array out(rank, m, n);
    for (long t = 0; t < rank; ++t)
      for (long i = 0; i < m; ++i)
        for (long j = 0; j < n; ++j)
          out(t, i, j) =
             nda::dcomplex{std::sin(.7 * t + .3 * i - .2 * j + phase), std::cos(.1 * t - .4 * i + .8 * j + phase)} / (rank * std::max(m, n));
    return out;
  }
  double difference(Array const &a, Array const &b) {
    if (a.shape() != b.shape()) throw std::runtime_error("Output shape mismatch");
    double error = 0;
    for (long i = 0; i < a.size(); ++i) {
      if (!std::isfinite(a.data()[i].real()) || !std::isfinite(a.data()[i].imag()) || !std::isfinite(b.data()[i].real())
          || !std::isfinite(b.data()[i].imag()))
        throw std::runtime_error("Nonfinite experiment result");
      error = std::max(error, std::abs(a.data()[i] - b.data()[i]));
    }
    return error;
  }
} // namespace
int main(int argc, char **argv) {
  try {
    long M = 4, K = 6, N = 3, samples = 3;
    double lambda = 40.;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (++i == argc) throw std::invalid_argument("Missing value");
      if (key == "--m")
        M = std::stol(argv[i]);
      else if (key == "--k")
        K = std::stol(argv[i]);
      else if (key == "--n")
        N = std::stol(argv[i]);
      else if (key == "--samples")
        samples = std::stol(argv[i]);
      else if (key == "--lambda")
        lambda = std::stod(argv[i]);
      else
        throw std::invalid_argument("Unknown argument " + key);
    }
    if (M < 1 || K < 1 || N < 1 || M > 252 || K > 252 || N > 252 || samples < 1 || samples > 5 || !std::isfinite(lambda) || lambda < 1
        || lambda > 120)
      throw std::invalid_argument("Beyond bounded primitive workload");
    cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, 1e-10, false), false);
    block_sparse_dlr_experiment::Hoisted hoisted(host);
    auto r  = host.rank();
    auto fc = coefficients(r, M, K, .1), gc = coefficients(r, K, N, .9);
    // Use the same cached LU on both CPU routes; this qualifies hoisting, not
    // the accuracy of recovering arbitrary coefficients from DLR values.
    auto fv = host.coefs2vals(fc), gv = host.coefs2vals(gc);
    auto recovered_f = host.vals2coefs(fv), recovered_g = host.vals2coefs(gv);
    auto reference      = host.convolve(2., fc, gc, cppdlr::TIME_ORDERED);
    auto edge_reference = host.convolve(2., recovered_f, recovered_g, cppdlr::TIME_ORDERED);
    std::cout << std::setprecision(17) << "backend,stage,rank,m,k,n,sample,call_ms,signal_norm,max_abs_error,relative_error,explicit_device_bytes\n";
    auto measure = [&](char const *backend, char const *stage, auto call, Array const &expected, std::size_t bytes = 0, bool exact = false) {
      auto warm   = call();
      double norm = 0;
      for (auto z : expected) norm = std::max(norm, std::abs(z));
      auto error = difference(warm, expected);
      if (exact ? error != 0. : error > 1e-10 * norm + 1e-14) throw std::runtime_error("Numerical control mismatch");
      for (long sample = 0; sample < samples; ++sample) {
        auto start  = Clock::now();
        auto result = call();
        auto ms     = elapsed(start);
        if (difference(result, warm) != 0.) throw std::runtime_error("Nondeterministic replay");
        std::cout << backend << ',' << stage << ',' << r << ',' << M << ',' << K << ',' << N << ',' << sample << ',' << ms << ',' << norm << ','
                  << error << ',' << (norm > 0 ? error / norm : 0.) << ',' << bytes << '\n';
      }
    };
    measure("cpu_original", "coefficients", [&] { return host.convolve(2., fc, gc, cppdlr::TIME_ORDERED); }, reference, 0, true);
    measure("cpu_hoisted", "coefficients", [&] { return hoisted.convolve(2., fc, gc); }, reference, 0, true);
    measure(
       "cpu_original", "edge", [&] { return host.convolve(2., host.vals2coefs(fv), host.vals2coefs(gv), cppdlr::TIME_ORDERED); }, edge_reference, 0,
       true);
    measure("cpu_hoisted", "edge", [&] { return hoisted.convolve(2., host.vals2coefs(fv), host.vals2coefs(gv)); }, edge_reference, 0, true);
#ifdef NDA_HAVE_CUDA
    namespace gpu = triqs_xca::block_sparse::cuda;
    auto owner    = std::make_shared<gpu::DeviceOps>();
    gpu::DlrOps device(owner, host);
    gpu::Tensor fd(fc), gd(gc), fvd(fv), gvd(gv), fs(r, M, K), gs(r, K, N), out(r, M, N);
    owner->wait_for_default_uploads();
    owner->finish();
    auto conv = [&] {
      device.convolve(2., fd, gd, out);
      owner->finish();
      return nda::to_host(out);
    };
    auto edge = [&] {
      device.vals2coefs(fvd, fs);
      device.vals2coefs(gvd, gs);
      device.convolve(2., fs, gs, out);
      owner->finish();
      return nda::to_host(out);
    };
    (void)conv();
    (void)edge(); // Materialize scratch before reporting retained bytes.
    auto payload = (fd.size() + gd.size() + fvd.size() + gvd.size() + fs.size() + gs.size() + out.size()) * sizeof(nda::dcomplex);
    measure("cuda_reference", "coefficients", conv, reference, device.storage_bytes() + payload);
    measure("cuda_reference", "edge", edge, edge_reference, device.storage_bytes() + payload);
#endif
    return 0;
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

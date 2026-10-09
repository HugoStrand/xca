// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
// Bounded C++ baseline for the dense CUDA port. Synthetic dense propagator,
// occupation-basis fermion operators, two poles, complex hybridization.
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include <triqs_xca/dense/cuda/profile_range.hpp>
#include <cppdlr/cppdlr.hpp>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>

using complex      = nda::dcomplex;
constexpr auto all = nda::range::all;
using clock_type   = std::chrono::steady_clock;

template <class F> double time_ms(F &&f, int repeats = 1) {
  auto start = clock_type::now();
  for (int i = 0; i < repeats; ++i) f();
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count() / repeats;
}

int main(int argc, char **argv) {
  try {
    int N           = argc > 1 ? std::stoi(argv[1]) : 16;
    int max_indices = argc > 2 ? std::stoi(argv[2]) : 8;
    int repeats     = argc > 3 ? std::stoi(argv[3]) : 3;
    if (N < 2 || N > 128 || !std::has_single_bit(static_cast<unsigned>(N)) || max_indices < 1 || max_indices > 100000 || repeats < 1
        || repeats > 1000)
      throw std::invalid_argument("usage: dense_cuda_benchmark [N=16 (power of 2, 2..128)] [max_indices=8] [repeats=3] [lambda=40] [api-only]");
    int n       = std::countr_zero(static_cast<unsigned>(N));
    double beta = 2., lambda = argc > 4 ? std::stod(argv[4]) : 40., eps = 1e-12;
    if (lambda < 5 || lambda > 200) throw std::invalid_argument("lambda must be in [5,200]");
    cppdlr::imtime_ops itops(lambda, cppdlr::build_dlr_rf(lambda, eps));
    int r = itops.rank();
    nda::vector<double> poles{-3., 1.2}; // beta-scaled, for the low-level constructor
    auto coeffs = nda::zeros<complex>(2, n, n);
    for (int p = 0; p < 2; ++p)
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) coeffs(p, i, j) = i == j ? complex{.4 + .1 * p} : complex{.025 * std::cos(i - j), .025 * std::sin(i - j)};
    auto Fs = nda::zeros<complex>(n, N, N), Fd = Fs;
    for (int f = 0; f < n; ++f)
      for (int state = 0; state < N; ++state)
        if (state & (1 << f)) Fs(f, state ^ (1 << f), state) = std::popcount(static_cast<unsigned>(state & ((1 << f) - 1))) % 2 ? -1. : 1.;
    for (int f = 0; f < n; ++f) Fd(f, all, all) = nda::conj(nda::transpose(Fs(f, all, all)));
    triqs_xca::dense::FSet fset(Fs, Fd, coeffs);
    auto start = clock_type::now();
    triqs_xca::dense::DiagramEvaluator evaluator(beta, eps, itops, poles, coeffs, fset);
    double construction = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    nda::matrix<complex> unitary(N, N), diagonal(N, N);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) unitary(i, j) = std::exp(complex{0., 2. * std::numbers::pi * i * j / N}) / std::sqrt(N);
    double partition = 0.;
    for (int i = 0; i < N; ++i) partition += std::exp(-beta * .6 * i / N);
    double eta = std::log(partition) / beta;
    auto g     = nda::zeros<complex>(r, N, N);
    for (int t = 0; t < r; ++t) {
      diagonal = 0.;
      for (int i = 0; i < N; ++i) diagonal(i, i) = -std::exp(-beta * cppdlr::rel2abs(itops.get_itnodes()(t)) * (.6 * i / N + eta));
      g(t, all, all) = nda::linalg::matmul(unitary, nda::linalg::matmul(diagonal, nda::conj(nda::transpose(unitary))));
    }
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G(std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(evaluator.tau_mesh, g)});
    std::cout << std::setprecision(10)
              << "backend,phase,N,r,n,p,order,backbones,first_ms,warm_ms,checksum,lambda,device_event_ms,owned_device_bytes,max_abs_error\n";
    auto row = [&](char const *phase, int order, long count, double first, double warm, double checksum, char const *backend = "cpu",
                   double event_ms = 0., std::size_t owned_bytes = 0, double max_error = 0.) {
      std::cout << backend << "," << phase << ',' << N << ',' << r << ',' << n << ",2," << order << ',' << count << ',' << first << ',' << warm << ','
                << checksum << ',' << lambda << ',' << event_ms << ',' << owned_bytes << ',' << max_error << '\n';
    };
    row("construction", 0, 0, construction, 0., 0.);
    auto work   = g;
    auto vertex = [&] {
      for (int t = 0; t < r; ++t) work(t, all, all) = nda::linalg::matmul(Fs(0, all, all), g(t, all, all));
    };
    double first = time_ms(vertex);
    row("left_vertex", 0, 0, first, time_ms(vertex, repeats), nda::max_element(nda::abs(work)));
    auto fc = itops.vals2coefs(g), gc = itops.vals2coefs(work);
    auto solve = [&] { work = itops.vals2coefs(g); };
    first      = time_ms(solve);
    row("vals2coefs", 0, 0, first, time_ms(solve, repeats), nda::max_element(nda::abs(work)));
    auto conv = [&] { work = itops.convolve(beta, fc, gc, cppdlr::TIME_ORDERED); };
    first     = time_ms(conv);
    row("convolution_current", 0, 0, first, time_ms(conv, repeats), nda::max_element(nda::abs(work)));
    auto reference = work;
    nda::matrix<double> H(r, r), Q(r, r);
    auto rf = itops.get_rfnodes(), it = itops.get_itnodes();
    auto C = itops.get_cf2it();
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < r; ++j) {
        H(i, j) = i == j ? 0. : cppdlr::k_it(0., rf(j)) / (rf(j) - rf(i));
        Q(i, j) = (it(i) > 0 ? it(i) : 1. + it(i)) * C(i, j) * cppdlr::k_it(0., rf(j));
      }
    auto hoisted = [&] {
      auto hf = cppdlr::arraymult(H, fc), hg = cppdlr::arraymult(H, gc);
      auto P = fc, R = fc;
      for (int t = 0; t < r; ++t) {
        P(t, all, all) = nda::linalg::matmul(fc(t, all, all), gc(t, all, all));
        R(t, all, all) = nda::linalg::matmul(hf(t, all, all), gc(t, all, all)) + nda::linalg::matmul(fc(t, all, all), hg(t, all, all));
      }
      work = beta * (cppdlr::arraymult(Q, P) + cppdlr::arraymult(C, R));
    };
    first                = time_ms(hoisted);
    double hoisted_error = nda::max_element(nda::abs(work - reference));
    if (!std::isfinite(hoisted_error) || hoisted_error > 1e-12)
      throw std::runtime_error("hoisted convolution differs from cppdlr: " + std::to_string(hoisted_error));
    row("convolution_hoisted_cpu", 0, 0, first, time_ms(hoisted, repeats), hoisted_error);
    auto reflection = [&] { work = itops.reflect(g); };
    first           = time_ms(reflection);
    row("reflection", 0, 0, first, time_ms(reflection, repeats), nda::max_element(nda::abs(work)));
#ifdef NDA_HAVE_CUDA
    using namespace triqs_xca::dense::cuda;
    // CUDA context initialization is explicitly separate from model construction.
    first = time_ms([&] { nda::device_error_check(cudaFree(nullptr), "CUDA context initialization"); });
    row("context", 0, 0, first, 0., 0., "cuda");
    auto gpu_start = clock_type::now();
    DiagramEvaluator gpu(beta, eps, itops, poles, coeffs, fset);
    row("construction", 0, 0, std::chrono::duration<double, std::milli>(clock_type::now() - gpu_start).count(), 0., 0., "cuda", 0.,
        gpu.workspace_bytes());
    if (argc <= 5 || std::string(argv[5]) != "api-only") {
      DlrOps dlr(itops, N);
      auto &ops = dlr.device_ops();
      tensor dg(g), dout(r, N, N), dcoeff(r, N, N), dfc(fc), dgc(gc);
      nda::cumatrix<complex> dF(Fs(0, all, all));
      cudaEvent_t begin = nullptr, end = nullptr;
      nda::device_error_check(cudaEventCreate(&begin), "create benchmark event");
      nda::device_error_check(cudaEventCreate(&end), "create benchmark event");
      auto event_measure = [&](auto &&fn) {
        nda::device_error_check(cudaEventRecord(begin, 0), "record start event");
        fn();
        nda::device_error_check(cudaEventRecord(end, 0), "record stop event");
        nda::device_error_check(cudaEventSynchronize(end), "synchronize benchmark event");
        float elapsed = 0.;
        nda::device_error_check(cudaEventElapsedTime(&elapsed, begin, end), "elapsed benchmark event");
        return double(elapsed);
      };
      auto phase = [&](char const *name, auto &&fn, auto const &expected, bool reconstruct = false) {
        double events     = 0.;
        double initial    = time_ms([&] { events = event_measure(fn); });
        events            = 0.;
        double warm       = time_ms([&] { events += event_measure(fn); }, repeats);
        double event_warm = events / repeats; // mean of subsequent calls
        if (reconstruct) dlr.coefs2vals(dout, dcoeff);
        auto host    = nda::to_host(reconstruct ? tensor_const_view{dcoeff} : tensor_const_view{dout});
        double error = nda::max_element(nda::abs(host - expected));
        if (!std::isfinite(error) || error > 1e-12) throw std::runtime_error(std::string("CUDA primitive differs: ") + name);
        row(name, 0, 0, initial, warm, nda::max_element(nda::abs(host)), "cuda", event_warm,
            dlr.workspace_bytes() + 5 * std::size_t(r) * N * N * sizeof(complex) + N * N * sizeof(complex), error);
      };
      auto vref = g;
      for (int t = 0; t < r; ++t) vref(t, all, all) = nda::linalg::matmul(Fs(0, all, all), g(t, all, all));
      phase("left_vertex", [&] { ops.left_product(dF, dg, dout); }, vref);
      phase("vals2coefs", [&] { dlr.vals2coefs(dg, dout); }, g, true);
      auto cref = itops.convolve(beta, fc, gc, cppdlr::TIME_ORDERED);
      phase("convolution", [&] { dlr.convolve(beta, dfc, dgc, dout); }, cref);
      auto rref = itops.reflect(g);
      phase("reflection", [&] { dlr.reflect(dg, dout); }, rref);
      nda::device_error_check(cudaEventDestroy(begin), "destroy benchmark event");
      nda::device_error_check(cudaEventDestroy(end), "destroy benchmark event");
    }
#endif
    std::vector<nda::array<int, 2>> topologies{{{0, 1}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 4}, {2, 5}}};
    for (int order = 1; order <= 3; ++order) {
      auto &topology = topologies[order - 1];
      long total = evaluator.get_num_self_energy_backbones(topology), count = std::min<long>(total, max_indices);
      nda::vector<long> indices(count);
      for (long i = 0; i < count; ++i) indices(i) = i * total / count;
      auto compute = [&] { work = evaluator.compute_self_energy(G, topology, indices)[0].data(); };
      first        = time_ms(compute);
      row("self_energy", order, count, first, time_ms(compute, repeats), nda::max_element(nda::abs(work)));
#ifdef NDA_HAVE_CUDA
      auto expected = work;
      auto run_gpu  = [&] { work = gpu.compute_self_energy(G, topology, indices)[0].data(); };
      first         = time_ms(run_gpu);
      double warm   = time_ms(run_gpu, repeats);
      double error  = nda::max_element(nda::abs(work - expected));
      if (!std::isfinite(error) || error > 1e-12) throw std::runtime_error("CUDA self-energy benchmark differs from CPU");
      row("self_energy", order, count, first, warm, nda::max_element(nda::abs(work)), "cuda", 0., gpu.workspace_bytes(), error);
      auto run_pairs         = [&] { work = gpu.compute_self_energy_by_pairs(G, topology, indices)[0].data(); };
      auto expected_pairs    = evaluator.compute_self_energy_by_pairs(G, topology, indices);
      auto run_cpu_pairs     = [&] { expected_pairs = evaluator.compute_self_energy_by_pairs(G, topology, indices); };
      double cpu_pairs_first = time_ms(run_cpu_pairs);
      double cpu_pairs_warm  = time_ms(run_cpu_pairs, repeats);
      row("self_energy_pairs", order, count, cpu_pairs_first, cpu_pairs_warm, nda::max_element(nda::abs(expected_pairs[0].data())));
      first = time_ms(run_pairs);
      warm  = time_ms(run_pairs, repeats);
      error = nda::max_element(nda::abs(work - expected_pairs[0].data()));
      if (!std::isfinite(error) || error > 1e-12) throw std::runtime_error("CUDA paired benchmark differs from CPU");
      row("self_energy_pairs", order, count, first, warm, nda::max_element(nda::abs(work)), "cuda", 0., gpu.workspace_bytes(), error);
      long corr_total = evaluator.get_num_single_ptcle_gf_backbones(topology);
      long corr_count = std::min<long>(corr_total, max_indices);
      nda::vector<long> corr_indices(corr_count);
      for (long i = 0; i < corr_count; ++i) corr_indices(i) = i * corr_total / corr_count;
      nda::array<complex, 3> corr;
      auto run_cpu_corr = [&] { corr = evaluator.compute_single_ptcle_gf(G, topology, corr_indices); };
      first             = time_ms(run_cpu_corr);
      warm              = time_ms(run_cpu_corr, repeats);
      row("single_particle", order, corr_count, first, warm, nda::max_element(nda::abs(corr)));
      auto expected_corr = corr;
      auto run_gpu_corr  = [&] { corr = gpu.compute_single_ptcle_gf(G, topology, corr_indices); };
      first              = time_ms(run_gpu_corr);
      warm               = time_ms(run_gpu_corr, repeats);
      error              = nda::max_element(nda::abs(corr - expected_corr));
      if (!std::isfinite(error) || error > 1e-12) throw std::runtime_error("CUDA correlator benchmark differs from CPU");
      row("single_particle", order, corr_count, first, warm, nda::max_element(nda::abs(corr)), "cuda", 0., gpu.workspace_bytes(), error);
#endif
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}

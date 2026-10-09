// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
// Bounded C++ baseline for the dense CUDA port. Synthetic dense propagator,
// occupation-basis fermion operators, two poles, complex hybridization.
#include <triqs_xca/dense/diagram_evaluator.hpp>
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
      throw std::invalid_argument("usage: dense_cuda_benchmark [N=16 (power of 2, 2..128)] [max_indices=8] [repeats=3]");
    int n       = std::countr_zero(static_cast<unsigned>(N));
    double beta = 2., lambda = 40., eps = 1e-12;
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
    std::cout << std::setprecision(10) << "backend,phase,N,r,n,p,order,backbones,first_ms,warm_ms,checksum\n";
    auto row = [&](char const *phase, int order, long count, double first, double warm, double checksum) {
      std::cout << "cpu," << phase << ',' << N << ',' << r << ',' << n << ",2," << order << ',' << count << ',' << first << ',' << warm << ','
                << checksum << '\n';
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
    first        = time_ms(hoisted);
    double error = nda::max_element(nda::abs(work - reference));
    if (error > 1e-12) throw std::runtime_error("hoisted convolution differs from cppdlr: " + std::to_string(error));
    row("convolution_hoisted_cpu", 0, 0, first, time_ms(hoisted, repeats), error);
    auto reflection = [&] { work = itops.reflect(g); };
    first           = time_ms(reflection);
    row("reflection", 0, 0, first, time_ms(reflection, repeats), nda::max_element(nda::abs(work)));
    std::vector<nda::array<int, 2>> topologies{{{0, 1}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 4}, {2, 5}}};
    for (int order = 1; order <= 3; ++order) {
      auto &topology = topologies[order - 1];
      long total = evaluator.get_num_self_energy_backbones(topology), count = std::min<long>(total, max_indices);
      nda::vector<long> indices(count);
      for (long i = 0; i < count; ++i) indices(i) = i * total / count;
      auto compute = [&] { work = evaluator.compute_self_energy(G, topology, indices)[0].data(); };
      first        = time_ms(compute);
      row("self_energy", order, count, first, time_ms(compute, repeats), nda::max_element(nda::abs(work)));
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}

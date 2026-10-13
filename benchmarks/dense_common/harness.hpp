// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <cppdlr/cppdlr.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>

// Shared deterministic workload and timing contract for dense evaluator benchmarks.
namespace dense_cuda_bench {
  using complex      = nda::dcomplex;
  using tensor       = nda::array<complex, 3>;
  using gf           = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>;
  constexpr auto all = nda::range::all;
  using clock        = std::chrono::steady_clock;
  inline double elapsed(clock::time_point t) { return std::chrono::duration<double, std::milli>(clock::now() - t).count(); }

  inline std::vector<std::string> split(std::string const &s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    for (std::string v; std::getline(in, v, ',');) {
      if (v.empty()) throw std::invalid_argument("empty list entry");
      out.push_back(v);
    }
    if (out.empty()) throw std::invalid_argument("empty list");
    return out;
  }
  inline int integer(std::string const &s, int lo, int hi) {
    std::size_t used = 0;
    long v           = std::stol(s, &used);
    if (used != s.size() || v < lo || v > hi) throw std::invalid_argument("integer out of range: " + s);
    return int(v);
  }
  struct Options {
    int N = 4, samples = 5, warmup = 1, seed = 1729;
    double lambda = 10., beta = 2.;
    std::vector<int> orders{1, 3, 6, 8}, counts{1, 8};
    std::vector<std::string> topologies{"crossing", "chain"}, methods{"se"};
    std::string backend = "both";
  };
  inline Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--help") {
        std::cout << "--dimension 4 [2,4,8,16,32,64] --orders 1,3,6,8 [1..12] --topologies crossing,chain\n"
                     "--indices 1,8 [1..4096] --methods se,pairs,spgf --samples 5 --warmup 1\n"
                     "--lambda 10 [5..120] --seed 1729 --backend both|cpu|candidate\n"
                     "CSV stdout; correctness failures exit nonzero. Candidate is CUDA in the baseline executable.\n";
        std::exit(0);
      }
      if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
      std::string v = argv[++i];
      auto ints     = [&](int lo, int hi) {
        std::vector<int> x;
        for (auto const &s : split(v)) x.push_back(integer(s, lo, hi));
        return x;
      };
      if (key == "--dimension")
        o.N = integer(v, 2, 64);
      else if (key == "--orders")
        o.orders = ints(1, 12);
      else if (key == "--indices")
        o.counts = ints(1, 4096);
      else if (key == "--samples")
        o.samples = integer(v, 1, 1000);
      else if (key == "--warmup")
        o.warmup = integer(v, 0, 100);
      else if (key == "--seed")
        o.seed = integer(v, 0, 2147483647);
      else if (key == "--lambda")
        o.lambda = integer(v, 5, 120);
      else if (key == "--topologies")
        o.topologies = split(v);
      else if (key == "--methods")
        o.methods = split(v);
      else if (key == "--backend")
        o.backend = v;
      else
        throw std::invalid_argument("unknown option: " + key);
    }
    if (!std::has_single_bit(unsigned(o.N))) throw std::invalid_argument("dimension must be a power of two");
    for (auto const &t : o.topologies)
      if (t != "crossing" && t != "chain") throw std::invalid_argument("unknown topology: " + t);
    for (auto const &m : o.methods)
      if (m != "se" && m != "pairs" && m != "spgf") throw std::invalid_argument("unknown method: " + m);
    if (o.backend != "both" && o.backend != "cpu" && o.backend != "candidate") throw std::invalid_argument("unknown backend");
    return o;
  }
  struct Fixture {
    Options options;
    int n;
    cppdlr::imtime_ops itops;
    nda::vector<double> poles{-3., 1.2}; // Dimensionless, low-level constructor convention.
    tensor coefficients, Fs, Fd;
    std::unique_ptr<triqs_xca::dense::FSet> fset;
    std::unique_ptr<triqs_xca::dense::DiagramEvaluator> cpu;
    gf G;
    double cpu_setup_ms = 0.;
    explicit Fixture(Options o)
       : options(o),
         n(std::countr_zero(unsigned(o.N))),
         itops(o.lambda, cppdlr::build_dlr_rf(o.lambda, 1e-12)),
         coefficients(2, n, n),
         Fs(n, o.N, o.N),
         Fd(n, o.N, o.N) {
      Fs = 0.;
      for (int p = 0; p < 2; ++p)
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j) coefficients(p, i, j) = i == j ? complex{.4 + .1 * p} : complex{.025 * std::cos(i - j), .025 * std::sin(i - j)};
      for (int f = 0; f < n; ++f)
        for (int s = 0; s < o.N; ++s)
          if (s & (1 << f)) Fs(f, s ^ (1 << f), s) = std::popcount(unsigned(s & ((1 << f) - 1))) % 2 ? -1. : 1.;
      for (int f = 0; f < n; ++f) Fd(f, all, all) = nda::conj(nda::transpose(Fs(f, all, all)));
      fset         = std::make_unique<triqs_xca::dense::FSet>(Fs, Fd, coefficients);
      auto begin   = clock::now();
      cpu          = std::make_unique<triqs_xca::dense::DiagramEvaluator>(o.beta, 1e-12, itops, poles, coefficients, *fset);
      cpu_setup_ms = elapsed(begin);
      nda::matrix<complex> U(o.N, o.N), D(o.N, o.N);
      for (int i = 0; i < o.N; ++i)
        for (int j = 0; j < o.N; ++j) U(i, j) = std::exp(complex{0., 2. * std::numbers::pi * i * j / o.N}) / std::sqrt(o.N);
      double Z = 0.;
      for (int i = 0; i < o.N; ++i) Z += std::exp(-o.beta * .6 * i / o.N);
      double eta = std::log(Z) / o.beta;
      tensor g(itops.rank(), o.N, o.N);
      for (int t = 0; t < itops.rank(); ++t) {
        D = 0.;
        for (int i = 0; i < o.N; ++i) D(i, i) = -std::exp(-o.beta * cppdlr::rel2abs(itops.get_itnodes()(t)) * (.6 * i / o.N + eta));
        g(t, all, all) = nda::linalg::matmul(U, nda::linalg::matmul(D, nda::conj(nda::transpose(U))));
      }
      G = gf(std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(cpu->tau_mesh, g)});
    }
  };
  inline nda::array<int, 2> topology(int order, std::string const &name) {
    nda::array<int, 2> t(order, 2);
    if (name == "crossing" || order == 1)
      for (int i = 0; i < order; ++i) {
        t(i, 0) = i;
        t(i, 1) = i + order;
      }
    else {
      t(0, 0) = 0;
      t(0, 1) = 2;
      for (int i = 1; i < order - 1; ++i) {
        t(i, 0) = 2 * i - 1;
        t(i, 1) = 2 * i + 2;
      }
      t(order - 1, 0) = 2 * order - 3;
      t(order - 1, 1) = 2 * order - 1;
    }
    return t;
  }
  inline nda::vector<long> select_indices(long total, int requested, int seed) {
    long count = std::min<long>(total, requested);
    nda::vector<long> out(count);
    std::mt19937_64 rng(seed);
    // One deterministic point in each disjoint stratum; no full enumeration.
    // Supported N/order bounds keep i*total below LONG_MAX.
    for (long i = 0; i < count; ++i) {
      long lo = i * total / count, hi = (i + 1) * total / count;
      out(i) = lo + long(rng() % std::uint64_t(hi - lo));
    }
    return out;
  }
  inline std::string encode(auto const &v) {
    std::ostringstream s;
    bool first = true;
    for (auto x : v) {
      if (!first) s << ':';
      s << x;
      first = false;
    }
    return s.str();
  }
  // An experiment implements exactly this narrow host-facing contract. Setup belongs
  // in its factory; count-independent lazy setup inside evaluate is charged to first call.
  struct Candidate {
    std::string name;
    std::function<tensor(std::string const &, nda::array<int, 2> const &, nda::vector<long> const &)> evaluate;
    std::function<std::size_t()> bytes;
  };
  inline tensor cpu_evaluate(Fixture &f, std::string const &method, nda::array<int, 2> const &t, nda::vector<long> const &ix) {
    if (method == "se") return tensor(f.cpu->compute_self_energy(f.G, t, ix)[0].data());
    if (method == "pairs") return tensor(f.cpu->compute_self_energy_by_pairs(f.G, t, ix)[0].data());
    return f.cpu->compute_single_ptcle_gf(f.G, t, ix);
  }
  struct Error {
    double norm = 0., absolute = 0., relative = 0.;
  };
  inline Error compare(tensor const &actual, tensor const &expected, bool enforce = true, char const *context = "comparison") {
    if (actual.shape() != expected.shape()) throw std::runtime_error("result shape differs");
    Error e;
    auto a = actual.begin(), b = expected.begin();
    for (; a != actual.end(); ++a, ++b) {
      if (!std::isfinite((*a).real()) || !std::isfinite((*a).imag()) || !std::isfinite((*b).real()) || !std::isfinite((*b).imag()))
        throw std::runtime_error("non-finite result");
      e.norm     = std::max(e.norm, std::abs(*b));
      e.absolute = std::max(e.absolute, std::abs(*a - *b));
    }
    e.relative = e.absolute / std::max(e.norm, 1e-300);
    if (!std::isfinite(e.norm) || !std::isfinite(e.absolute) || !std::isfinite(e.relative))
      throw std::runtime_error("non-finite comparison metric");
    // A fixed absolute tolerance would accept zero instead of tiny high-order
    // contributions. Require relative parity with the reference evaluator.
    // The underflow floor is far below every supported benchmark signal.
    // This does not replace the existing analytic/unit tests.
    if (enforce && e.absolute > 1e-280 + 1e-10 * e.norm) {
      std::ostringstream msg;
      msg << context << std::scientific << " failed: norm=" << e.norm << " abs=" << e.absolute << " rel=" << e.relative;
      throw std::runtime_error(msg.str());
    }
    return e;
  }
  struct Runtime {
    std::function<void()> synchronize = [] {};
    std::function<double()> initialize = [] { return 0.; };
    std::function<std::shared_ptr<void>(bool)> profile = [](bool) { return std::shared_ptr<void>{}; };
  };
  inline int run_workload(int argc, char **argv, std::function<Candidate(Fixture &)> factory = {},
                          std::function<Candidate(Fixture &)> reference_factory = {}, Runtime runtime = {}) {
    try {
      auto o = parse(argc, argv);
      Fixture f(o);
      std::unique_ptr<Candidate> candidate;
      std::unique_ptr<Candidate> reference;
      double setup = 0., context = 0.;
      if (o.backend != "cpu") {
        if (!factory || !reference_factory) throw std::invalid_argument("candidate and reference factories are required");
        context = runtime.initialize();
        auto begin = clock::now();
        candidate  = std::make_unique<Candidate>(factory(f));
        runtime.synchronize();
        setup     = elapsed(begin);
        reference = std::make_unique<Candidate>(reference_factory(f));
      }
      std::cout
         << std::setprecision(17)
         << "fixture,backend,N,r,n,p,beta,lambda,order,topology,method,requested_indices,indices,total_indices,index_list,sample,wall_ms,ns_per_index,setup_ms,context_ms,owned_bytes,result_norm,max_abs_error,relative_error,cpu_abs_error,cpu_relative_error,index_cpu_abs_error,index_cpu_relative_error,seed\n";
      for (int order : o.orders)
        for (auto const &name : o.topologies) {
          if (name == "chain" && order <= 2 && std::find(o.topologies.begin(), o.topologies.end(), "crossing") != o.topologies.end()) continue;
          auto t = topology(order, name);
          for (auto const &method : o.methods)
            for (int requested : o.counts) {
              long total           = method == "spgf" ? f.cpu->get_num_single_ptcle_gf_backbones(t) : f.cpu->get_num_self_energy_backbones(t);
              auto ix              = select_indices(total, requested, o.seed);
              tensor cpu_expected  = cpu_evaluate(f, method, t, ix);
              tensor expected      = reference ? reference->evaluate(method, t, ix) : cpu_expected;
              double index_cpu_abs = 0., index_cpu_rel = 0.;
              // Validate each selected index, preventing small/zero sums from hiding wrong lanes.
              if (candidate)
                for (long index : ix) {
                  nda::vector<long> one{index};
                  auto gold = reference->evaluate(method, t, one);
                  compare(candidate->evaluate(method, t, one), gold, true, "candidate/CUDA individual");
                  auto ce       = compare(gold, cpu_evaluate(f, method, t, one), order <= 3, "baseline CUDA/CPU individual");
                  index_cpu_abs = std::max(index_cpu_abs, ce.absolute);
                  index_cpu_rel = std::max(index_cpu_rel, ce.relative);
                }
              if (candidate) {
                // Guard against stale-input/index caches before trusting a speedup.
                nda::vector<long> duplicates{ix(0), ix(0), ix(ix.size() - 1)}, empty(0);
                compare(candidate->evaluate(method, t, duplicates), reference->evaluate(method, t, duplicates));
                compare(candidate->evaluate(method, t, empty), reference->evaluate(method, t, empty));
                tensor saved = f.G[0].data();
                f.G[0].data() *= complex{.99, .001};
                compare(candidate->evaluate(method, t, ix), reference->evaluate(method, t, ix));
                f.G[0].data()() = saved;
              }
              // Validation may populate graph/schedule caches. Fresh measured
              // objects charge their own construction and lazy preparation.
              f.cpu.reset();
              auto cpu_begin = clock::now();
              f.cpu          = std::make_unique<triqs_xca::dense::DiagramEvaluator>(o.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
              f.cpu_setup_ms = elapsed(cpu_begin);
              if (candidate) {
                candidate.reset();
                auto candidate_begin = clock::now();
                candidate            = std::make_unique<Candidate>(factory(f));
                runtime.synchronize();
                setup = elapsed(candidate_begin);
              }
              auto measure = [&](bool gpu, int sample) {
                auto timed_range = runtime.profile(gpu);
                if (gpu) runtime.synchronize();
                auto begin  = clock::now();
                auto result = gpu ? candidate->evaluate(method, t, ix) : cpu_evaluate(f, method, t, ix);
                if (gpu) runtime.synchronize();
                double ms = elapsed(begin);
                auto e    = compare(result, gpu ? expected : cpu_expected, true, "timed result/reference");
                auto ce   = compare(result, cpu_expected, gpu && order <= 3, "timed result/CPU");
                if (sample >= -1)
                  std::cout << "dense-small-v2," << (gpu ? candidate->name : "cpu") << ',' << o.N << ',' << f.itops.rank() << ',' << f.n << ",2,"
                            << o.beta << ',' << o.lambda << ',' << order << ',' << name << ',' << method << ',' << requested << ',' << ix.size()
                            << ',' << total << ',' << encode(ix) << ',' << sample << ',' << ms << ',' << ms * 1e6 / ix.size() << ','
                            << (gpu ? setup : f.cpu_setup_ms) << ',' << (gpu ? context : 0.) << ',' << (gpu ? candidate->bytes() : 0) << ',' << e.norm
                            << ',' << e.absolute << ',' << e.relative << ',' << ce.absolute << ',' << ce.relative << ',' << index_cpu_abs << ','
                            << index_cpu_rel << ',' << o.seed << '\n';
              };
              // sample -1 is first full-vector measurement after correctness/setup; -2 is unreported warmup.
              bool do_cpu = o.backend != "candidate";
              if (do_cpu) measure(false, -1);
              if (candidate) measure(true, -1);
              for (int j = 0; j < o.warmup; ++j) {
                if (do_cpu) measure(false, -2);
                if (candidate) measure(true, -2);
              }
              for (int j = 0; j < o.samples; ++j) {
                // Alternate order to reduce monotonic thermal/drift bias.
                if (j % 2 && candidate) measure(true, j);
                if (do_cpu) measure(false, j);
                if (j % 2 == 0 && candidate) measure(true, j);
              }
              std::cout.flush();
            }
        }
      return 0;
    } catch (std::exception const &e) {
      std::cerr << "dense benchmark: " << e.what() << '\n';
      return 2;
    }
  }
} // namespace dense_cuda_bench

#pragma once
#include "convolution_experiment.hpp"
#include <chrono>
#include <map>
#include <string>

namespace small_n_probe {
  using Clock = std::chrono::steady_clock;
  struct Counter {
    double us = 0;
    long calls = 0;
  };
  inline bool profiling = false, fixed_vertices = false, production_convolution = false;
  inline small_n::Policy policy = small_n::Policy::complex_blas;
  inline small_n::Convolution const *convolution = nullptr;
  inline std::map<std::string, Counter> counters;

  struct Scope {
    char const *name;
    Clock::time_point start;
    explicit Scope(char const *label) : name(profiling ? label : nullptr) {
      if (name) start = Clock::now();
    }
    ~Scope() {
      if (name) {
        auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        auto &entry = counters[name];
        entry.us += elapsed;
        ++entry.calls;
      }
    }
  };

  template <typename... Args> void convolve(cppdlr::fast_convolve const &baseline, Args &&...args) {
    Scope scope("convolution");
    if (production_convolution) baseline.convolve(std::forward<Args>(args)...);
    else convolution->convolve(std::forward<Args>(args)..., policy);
  }

  template <typename Alpha, typename A, typename B, typename Beta, typename C>
  void gemm(Alpha alpha, A const &a, B const &b, Beta beta, C &&c) {
    if (fixed_vertices) small_n::gemm4(alpha, a, b, beta, c);
    else nda::blas::gemm(alpha, a, b, beta, c);
  }
} // namespace small_n_probe

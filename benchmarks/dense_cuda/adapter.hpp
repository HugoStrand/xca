// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../dense_common/harness.hpp"
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include <triqs_xca/dense/cuda/profile_range.hpp>

namespace dense_cuda_bench {
  inline Candidate baseline(Fixture &f) {
#ifdef NDA_HAVE_CUDA
    auto p = std::make_shared<triqs_xca::dense::cuda::DiagramEvaluator>(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
    return {"cuda_baseline",
            [p, &f](auto const &m, auto const &t, auto const &ix) -> tensor {
              if (m == "se") return tensor(p->compute_self_energy(f.G, t, ix)[0].data());
              if (m == "pairs") return tensor(p->compute_self_energy_by_pairs(f.G, t, ix)[0].data());
              return p->compute_single_ptcle_gf(f.G, t, ix);
            },
            [p] { return p->workspace_bytes(); }};
#else
    (void)f;
    throw std::runtime_error("candidate requires a CUDA-enabled build; use --backend cpu");
#endif
  }
  inline Runtime cuda_runtime() {
    Runtime runtime;
#ifdef NDA_HAVE_CUDA
    runtime.synchronize = [] { nda::device_error_check(cudaDeviceSynchronize(), "benchmark API completion"); };
    runtime.initialize = [] {
      auto begin = clock::now();
      nda::device_error_check(cudaFree(nullptr), "benchmark context");
      return elapsed(begin);
    };
    runtime.profile = [](bool gpu) -> std::shared_ptr<void> {
      return std::make_shared<triqs_xca::dense::cuda::ProfileRange>(gpu ? "bench.cuda.sample" : "bench.cpu.sample");
    };
#endif
    return runtime;
  }
  inline int run(int argc, char **argv, std::function<Candidate(Fixture &)> factory = baseline) {
    return run_workload(argc, argv, std::move(factory), baseline, cuda_runtime());
  }
} // namespace dense_cuda_bench

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../../../benchmarks/dense_cuda_small/harness.hpp"
#include "diagram_evaluator.hpp"
int main(int argc, char **argv) {
  return dense_cuda_bench::run(argc, argv, [](dense_cuda_bench::Fixture &f) {
    auto p = std::make_shared<xca_experiments::graphs::DiagramEvaluator>(
      f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
    return dense_cuda_bench::Candidate{"graphs", [p,&f](auto const &m, auto const &t, auto const &ix) {
      if (m != "se") throw std::invalid_argument("graphs supports ordinary SE only");
      return dense_cuda_bench::tensor(p->compute_self_energy(f.G,t,ix)[0].data());
    }, [p] { return p->workspace_bytes(); }};
  });
}

// Test-only adapter: experimental ordinary SE; baseline pairs/SPGF/correlators.
// Reuses existing multi-evaluator analytic tests without claiming unsupported paths.
#pragma once
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include "diagram_evaluator.hpp"
class StreamTestAdapter : public triqs_xca::dense::cuda::DiagramEvaluator {
  xca_experiments::serial_stream::DiagramEvaluator candidate_;
public:
  template<class... Args> explicit StreamTestAdapter(Args&&... args)
    : triqs_xca::dense::cuda::DiagramEvaluator(args...), candidate_(args...) {}
  template<class... Args> auto compute_self_energy(Args&&... args) {
    return candidate_.compute_self_energy(std::forward<Args>(args)...);
  }
};

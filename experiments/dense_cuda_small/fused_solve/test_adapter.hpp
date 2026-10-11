#pragma once
#include "diagram_evaluator.hpp"
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
// Experimental ordinary SE; baseline for unsupported pairs/SPGF/correlators.
class FusedTestAdapter : public triqs_xca::dense::cuda::DiagramEvaluator {
  xca_experiments::fused_solve::DiagramEvaluator candidate_;

public:
  template <class... Args>
  explicit FusedTestAdapter(Args &&...args)
      : triqs_xca::dense::cuda::DiagramEvaluator(args...), candidate_(args...) {
  }
  template <class... Args> auto compute_self_energy(Args &&...args) {
    return candidate_.compute_self_energy(std::forward<Args>(args)...);
  }
};

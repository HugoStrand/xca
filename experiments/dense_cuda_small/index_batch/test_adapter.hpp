#pragma once
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include "diagram_evaluator.hpp"
// Experimental ordinary SE; baseline for unsupported pairs/SPGF/correlators.
class BatchTestAdapter : public triqs_xca::dense::cuda::DiagramEvaluator {
  xca_experiments::index_batch::DiagramEvaluator candidate_;

  public:
  template <class... Args> explicit BatchTestAdapter(Args &&...args) : triqs_xca::dense::cuda::DiagramEvaluator(args...), candidate_(args...) {}
  template <class... Args> auto compute_self_energy(Args &&...args) { return candidate_.compute_self_energy(std::forward<Args>(args)...); }
};

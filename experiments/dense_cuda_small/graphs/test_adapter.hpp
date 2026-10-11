#pragma once
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include "diagram_evaluator.hpp"
// Existing suites enumerate at most128 ordinary-SE indices through order3.
// Other methods remain baseline tests, with no experimental support claimed.
class GraphTestAdapter : public triqs_xca::dense::cuda::DiagramEvaluator {
  xca_experiments::graphs::DiagramEvaluator candidate_;
public:
  template<class... Args> explicit GraphTestAdapter(Args&&... args)
    : triqs_xca::dense::cuda::DiagramEvaluator(args...), candidate_(args...) {}
  template<class G,class T> auto compute_self_energy(G&& g,T&& t) {
    auto n=candidate_.get_num_self_energy_backbones(t);
    if(n>128) throw std::invalid_argument("graph analytic test exceeds bounded scope");
    nda::vector<long> ix(n); for(long i=0;i<n;++i) ix(i)=i;
    return candidate_.compute_self_energy(g,t,ix);
  }
  template<class G,class T,class I> auto compute_self_energy(G&& g,T&& t,I&& i) {
    return candidate_.compute_self_energy(g,t,std::forward<I>(i));
  }
};

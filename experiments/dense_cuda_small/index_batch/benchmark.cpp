#include "../../../benchmarks/dense_cuda_small/harness.hpp"
#include "diagram_evaluator.hpp"
#include <cstdlib>
int main(int argc, char **argv) {
  return dense_cuda_bench::run(argc, argv, [](dense_cuda_bench::Fixture &f) {
    int b = 8;
    if (auto value = std::getenv("XCA_BATCH_CAPACITY")) b = dense_cuda_bench::integer(value, 1, 128);
    auto p = std::make_shared<xca_experiments::index_batch::DiagramEvaluator>(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset, b);
    bool cache = true;
    if (auto value = std::getenv("XCA_BATCH_CACHE")) cache = dense_cuda_bench::integer(value, 0, 1);
    p->set_cache_enabled(cache);
    return dense_cuda_bench::Candidate{"index_batch_b" + std::to_string(b) + (cache ? "" : "_uncached"),
                                       [p, &f](auto const &m, auto const &t, auto const &ix) {
                                         if (m != "se") throw std::invalid_argument("index_batch supports ordinary SE only");
                                         return dense_cuda_bench::tensor(p->compute_self_energy(f.G, t, ix)[0].data());
                                       },
                                       [p] { return p->workspace_bytes(); }};
  });
}

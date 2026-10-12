#include "benchmarks/dense_cuda_small/harness.hpp"
#include "diagram_evaluator.hpp"
#include <cstdlib>
#include <climits>

int main(int argc, char **argv) {
  return dense_cuda_bench::run(argc, argv, [](dense_cuda_bench::Fixture &f) -> dense_cuda_bench::Candidate {
    int capacity = 128;
    if (auto value = std::getenv("XCA_BATCH_CAPACITY")) capacity = dense_cuda_bench::integer(value, 1, INT_MAX);
    auto p = std::make_shared<triqs_xca::dense::cuda_small_probe::DiagramEvaluator>(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
    p->set_batch_capacity(capacity);
    return {"native_rank_probe_b" + std::to_string(capacity),
      [p, &f](auto const &method, auto const &topology, auto const &indices) -> dense_cuda_bench::tensor {
        if (method == "se") return dense_cuda_bench::tensor(p->compute_self_energy(f.G, topology, indices)[0].data());
        if (method == "pairs") return dense_cuda_bench::tensor(p->compute_self_energy_by_pairs(f.G, topology, indices)[0].data());
        return p->compute_single_ptcle_gf(f.G, topology, indices);
      }, [p] { return p->workspace_bytes(); }};
  });
}

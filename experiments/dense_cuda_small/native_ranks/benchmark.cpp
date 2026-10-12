#include "benchmarks/dense_cuda_small/harness.hpp"
#include "diagram_evaluator.hpp"
#include "probe_config.hpp"
#include <cstdlib>
#include <climits>

int main(int argc, char **argv) {
  return dense_cuda_bench::run(argc, argv, [](dense_cuda_bench::Fixture &f) -> dense_cuda_bench::Candidate {
    int capacity = 128;
    if (auto value = std::getenv("XCA_BATCH_CAPACITY")) capacity = dense_cuda_bench::integer(value, 1, INT_MAX);
    auto p = std::make_shared<triqs_xca::dense::cuda_small_probe::DiagramEvaluator>(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
    p->set_batch_capacity(capacity);
    return {std::string(XCA_PROBE_PRODUCTS ? "native_rank_tile4_b" : "native_rank_library_b") + std::to_string(capacity),
      [p, &f, capacity](auto const &method, auto const &topology, auto const &indices) -> dense_cuda_bench::tensor {
        if (method == "se" || method == "pairs") {
          auto before=p->batch_chunks();
          auto result=method == "se" ? p->compute_self_energy(f.G,topology,indices) : p->compute_self_energy_by_pairs(f.G,topology,indices);
          if (indices.size()>=8 && capacity>=8 && (f.options.N==4 || f.options.N==8) && p->batch_chunks()==before)
            throw std::runtime_error("Rank probe did not exercise native batching");
          return dense_cuda_bench::tensor(result[0].data());
        }
        return p->compute_single_ptcle_gf(f.G, topology, indices);
      }, [p] { return p->workspace_bytes(); }};
  });
}

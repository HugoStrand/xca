#include "../block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<gpu::DiagramEvaluator>(
     argc, argv, "cuda_reference", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<gpu::DiagramEvaluator>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh);
     });
}

#include "cpu_scheduler.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
  return block_sparse_bench::run<scalar_path_experiment::CpuScalarPathEvaluator>(
     argc, argv, "cpu_scalar_paths", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<scalar_path_experiment::CpuScalarPathEvaluator>(poles, coeff, mesh, ad, ops, dyn);
     });
}

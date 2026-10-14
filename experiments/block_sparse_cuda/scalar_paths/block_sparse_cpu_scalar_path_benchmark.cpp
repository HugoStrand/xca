#include "cpu_scheduler.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
int main(int argc, char **argv) {
#ifdef XCA_EXPERIMENT_SCALAR_CONSTANTS
  constexpr auto backend = "cpu_scalar_constants";
  constexpr bool exact_control = true;
#else
  constexpr auto backend = "cpu_scalar_paths";
  constexpr bool exact_control = false;
#endif
  return block_sparse_bench::run<scalar_path_experiment::CpuScalarPathEvaluator>(
     argc, argv, backend, [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<scalar_path_experiment::CpuScalarPathEvaluator>(poles, coeff, mesh, ad, ops, dyn);
     }, exact_control);
}

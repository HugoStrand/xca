#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#ifdef XCA_DENSITY_SETS_CPU
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#elif defined(XCA_DENSITY_SETS_GENERAL)
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#else
#include "scalar_evaluator.hpp"
#endif

int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  auto fixture = [](auto const &options, auto &operators, auto &dyn) {
    if (!options.dynamic || options.partition != "occupation" || options.observable != "se")
      throw std::invalid_argument("Density-set benchmark requires --dynamic --partition occupation --observable se");
    operators.clear();
    long width = std::min(3, options.modes);
    for (int i = 0; i < width; ++i) operators.push_back(triqs::operators::n("f", i) + .01 * (i + 1));
    dyn = nda::zeros<nda::dcomplex>(2, width, width);
    for (long p = 0; p < 2; ++p)
      for (long i = 0; i < width; ++i)
        for (long j = 0; j < width; ++j) dyn(p, i, j) = nda::dcomplex{.03 * (1 + i + j), .01 * (i - j)} / double(p + 1);
  };
  auto validate = [](auto const &model) {
    if (!std::all_of(model.dimensions().begin(), model.dimensions().end(), [](long d) { return d == 1; })
        || !std::any_of(model.sets().begin(), model.sets().end(), [](auto const &set) { return set.size() > 1; }))
      throw std::runtime_error("Density-set fixture did not produce scalar blocks and a multi-field set");
  };
#ifdef XCA_DENSITY_SETS_CPU
  using Evaluator = triqs_xca::block_sparse::DiagramEvaluator;
  constexpr auto backend = "cpu_density_sets";
#elif defined(XCA_DENSITY_SETS_GENERAL)
  using Evaluator = gpu::DiagramEvaluator;
  constexpr auto backend = "cuda_general_density_sets";
#else
  using Evaluator = gpu::ScalarPathEvaluator;
  constexpr auto backend = "cuda_batched_density_sets";
#endif
  return block_sparse_bench::run<Evaluator>(argc, argv, backend,
    [&](auto const &ad, auto coeff, auto mesh, auto poles, auto operators, auto dyn) {
      auto model = gpu::export_model(ad, coeff, operators, dyn);
      validate(model);
#ifdef XCA_DENSITY_SETS_CPU
      return std::make_unique<Evaluator>(poles, coeff, mesh, ad, operators, dyn);
#else
      return std::make_unique<Evaluator>(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Ordered);
#endif
    }, false, fixture);
}

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "scalar_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <cuda_profiler_api.h>
class ProfileScalar : public triqs_xca::block_sparse::cuda::ScalarPathEvaluator {
  long calls = 0;

  public:
  using ScalarPathEvaluator::ScalarPathEvaluator;
  auto compute_self_energy_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, nda::array_const_view<int, 2> topology, long first,
                                 long last) {
    bool capture = ++calls == 2; // The common runner's first call is warm-up.
    if (capture && cudaProfilerStart() != cudaSuccess) throw std::runtime_error("Cannot start scoped profiler");
    try {
      auto result = ScalarPathEvaluator::compute_self_energy_range(G, topology, first, last);
      if (capture && cudaProfilerStop() != cudaSuccess) throw std::runtime_error("Cannot stop scoped profiler");
      return result;
    } catch (...) {
      if (capture) cudaProfilerStop();
      throw;
    }
  }
};
int main(int argc, char **argv) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  return block_sparse_bench::run<ProfileScalar>(
     argc, argv, "profile_scalar_fanin", [](auto const &ad, auto coeff, auto mesh, auto poles, auto ops, auto dyn) {
       return std::make_unique<ProfileScalar>(gpu::export_model(ad, coeff, ops, dyn), poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024,
                                              gpu::SparseBackend::Ordered);
     });
}

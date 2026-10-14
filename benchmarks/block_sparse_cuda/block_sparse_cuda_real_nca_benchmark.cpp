// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../block_sparse_common/call_runner.hpp"
namespace gpu = triqs_xca::block_sparse::cuda;
struct CheckedRealNca : gpu::DiagramEvaluator {
  bool expect_fast;
  CheckedRealNca(gpu::OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh, bool expected)
    : gpu::DiagramEvaluator(std::move(model),poles,std::move(mesh),1ULL<<30,32ULL<<20,gpu::SparseBackend::Ordered,gpu::SpecialBackend::Reference,gpu::OutputPlacement::Resident,512ULL<<20,gpu::NcaBackend::RealBatch),expect_fast(expected) {}
  auto compute_self_energy_range(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G,nda::array_const_view<int,2> topology,long first,long last) {
    auto result=gpu::DiagramEvaluator::compute_self_energy_range(G,topology,first,last);
    if ((last_nca_batch_paths()>0)!=expect_fast) throw std::runtime_error("Unexpected real NCA dispatch");
    return result;
  }
};
int main(int argc,char **argv) {
  auto options=block_sparse_bench::options(argc,argv);
  bool expected=options.partition=="occupation" && options.order==1;
  return block_sparse_bench::run<CheckedRealNca>(argc,argv,"cuda_real_nca_batch",
    [&](auto const &ad,auto coeff,auto mesh,auto poles,auto ops,auto dyn) {
      return std::make_unique<CheckedRealNca>(gpu::export_model(ad,coeff,ops,dyn),poles,mesh,expected);
    });
}

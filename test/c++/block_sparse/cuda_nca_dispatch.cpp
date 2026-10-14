// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/real_nca.hpp>
namespace gpu = triqs_xca::block_sparse::cuda;
namespace {
class BatchEvaluator : public gpu::DiagramEvaluator {
public:
  BatchEvaluator(gpu::OperatorModel model, nda::vector_const_view<double> poles, triqs::mesh::dlr_imtime mesh,
                 std::size_t budget=1ULL<<30, std::size_t sparse=32ULL<<20,
                 gpu::SparseBackend backend=gpu::SparseBackend::Library,
                 gpu::SpecialBackend special=gpu::SpecialBackend::Reference,
                 gpu::OutputPlacement placement=gpu::OutputPlacement::Resident)
    :gpu::DiagramEvaluator(std::move(model),poles,std::move(mesh),budget,sparse,backend,special,placement,512ULL<<20,gpu::NcaBackend::RealBatch){}
};

void exact(auto const &a, auto const &b) {
  ASSERT_EQ(a.size(), b.size());
  for (long i=0;i<a.size();++i) EXPECT_EQ(nda::max_element(nda::abs(a[i].data()-b[i].data())),0.);
}
}
TEST(ProductionRealNca, MultiSetTailChangedInputsAndGeneralFallback) {
  if (!gpu::real_nca_available()) GTEST_SKIP();
  auto ad=block_sparse_bench::atom(8,"occupation");
  auto coefficients=block_sparse_bench::coefficients(ad);
  std::vector<triqs::operators::many_body_operator_complex> operators{triqs::operators::n("f",0)+.01,triqs::operators::n("f",1)+.02};
  auto dyn=nda::zeros<nda::dcomplex>(2,2,2);
  for (long p=0;p<2;++p) for (long i=0;i<2;++i) for (long j=0;j<2;++j) dyn(p,i,j)=.03*(1+i+j)/(p+1);
  auto model=gpu::export_model(ad,coefficients,operators,dyn);
  triqs::mesh::dlr_imtime mesh(2.,triqs::mesh::Fermion,20.,1e-10,false);
  nda::vector<double> poles{-1.5,.6};
  auto G=block_sparse_bench::propagator(ad,mesh);
  nda::array<int,2> nca{{0,1}},oca{{0,2},{1,3}};
  for (auto sparse:{gpu::SparseBackend::Library,gpu::SparseBackend::Ordered})
    for (auto special:{gpu::SpecialBackend::Reference,gpu::SpecialBackend::Mapped}) {
      gpu::DiagramEvaluator reference(model,poles,mesh,1ULL<<30,32ULL<<20,sparse,special);
      BatchEvaluator candidate(model,poles,mesh,1ULL<<30,32ULL<<20,sparse,special);
      for (int update=0;update<3;++update) {
        for (long b=0;b<G.size();++b) G[b].data()=nda::dcomplex{.2+.001*b,update==1?.01:0.};
        G[7].data()=0.;
        auto expected=reference.compute_self_energy(G,nca);
        auto actual=candidate.compute_self_energy(G,nca);
        exact(actual,expected);
        EXPECT_EQ(candidate.last_contributing_paths(),reference.last_contributing_paths());
        if (update==1) EXPECT_EQ(candidate.last_nca_batch_paths(),0);
        else { EXPECT_GT(candidate.last_nca_batch_paths(),512); EXPECT_NE(candidate.last_nca_batch_paths()%512,0); }
      }
      exact(candidate.compute_self_energy_range(G,oca,0,1),reference.compute_self_energy_range(G,oca,0,1));
      EXPECT_EQ(candidate.last_nca_batch_paths(),0);
      BatchEvaluator streamed(model,poles,mesh,1ULL<<30,32ULL<<20,sparse,special,gpu::OutputPlacement::HostSectors);
      exact(streamed.compute_self_energy(G,nca),reference.compute_self_energy(G,nca));
      EXPECT_EQ(streamed.last_nca_batch_paths(),0);
    }
}
TEST(ProductionRealNca, AdjacentCutoffWithoutReplay) {
  if (!gpu::real_nca_available()) GTEST_SKIP();
  auto ad=block_sparse_bench::atom(2,"occupation");
  auto coefficients=nda::zeros<nda::dcomplex>(1,2,2);coefficients(0,0,0)=coefficients(0,1,1)=1.;
  auto model=gpu::export_model(ad,coefficients);
  triqs::mesh::dlr_imtime mesh(2.,triqs::mesh::Fermion,20.,1e-10,false);
  nda::vector<double> poles{0.};nda::array<int,2> nca{{0,1}};
  auto G=block_sparse_bench::propagator(ad,mesh);
  gpu::DiagramEvaluator reference(model,poles,mesh,1ULL<<30,32ULL<<20,gpu::SparseBackend::Ordered);
  BatchEvaluator candidate(model,poles,mesh);
  for (double amplitude:{std::nextafter(2e-16,0.),2e-16,std::nextafter(2e-16,1.),1e-310,1e100}) {
    for (long b=0;b<G.size();++b) G[b].data()=0.;G[0].data()=amplitude;
    exact(candidate.compute_self_energy(G,nca),reference.compute_self_energy(G,nca));
    EXPECT_EQ(candidate.last_contributing_paths(),reference.last_contributing_paths());
  }
  for (long b=0;b<G.size();++b) G[b].data()=0.;
  G[0].data()=1.; G[0].data()(0,0,0)=1e-310;
  exact(candidate.compute_self_energy(G,nca),reference.compute_self_energy(G,nca));
  EXPECT_GT(candidate.last_nca_batch_paths(),0);
}
TEST(ProductionRealNca, UnsafeRangeAndComplexModelFallback) {
  if (!gpu::real_nca_available()) GTEST_SKIP();
  auto ad=block_sparse_bench::atom(2,"occupation");
  auto coefficients=nda::zeros<nda::dcomplex>(1,2,2);coefficients(0,0,0)=coefficients(0,1,1)=1e200;
  triqs::mesh::dlr_imtime mesh(2.,triqs::mesh::Fermion,20.,1e-10,false);
  nda::vector<double> poles{0.};nda::array<int,2> nca{{0,1}};
  auto G=block_sparse_bench::propagator(ad,mesh);
  BatchEvaluator candidate(gpu::export_model(ad,coefficients),poles,mesh);
  for (long b=0;b<G.size();++b) G[b].data()=1e200;
  candidate.compute_self_energy(G,nca);EXPECT_EQ(candidate.last_nca_batch_paths(),0);
  for (long b=0;b<G.size();++b) G[b].data()=1.;
  candidate.compute_self_energy(G,nca);EXPECT_GT(candidate.last_nca_batch_paths(),0);
  coefficients(0,0,0)=nda::dcomplex{.3,.01};coefficients(0,1,1)=.3;
  auto model=gpu::export_model(ad,coefficients);
  BatchEvaluator complex(model,poles,mesh);
  gpu::DiagramEvaluator reference(model,poles,mesh);
  exact(complex.compute_self_energy(G,nca),reference.compute_self_energy(G,nca));
  EXPECT_EQ(complex.last_nca_batch_paths(),0);
}

TEST(ProductionRealNca, RangeAdmissionAndOtherObservables) {
  if (!gpu::real_nca_available()) GTEST_SKIP();
  triqs::mesh::dlr_imtime mesh(2.,triqs::mesh::Fermion,20.,1e-10,false);
  nda::vector<double> poles{-1.5,.6}; nda::array<int,2> nca{{0,1}};
  for (auto partition : {"occupation", "number", "mixed"}) {
    auto ad=block_sparse_bench::atom(4,partition);
    auto model=gpu::export_model(ad,block_sparse_bench::coefficients(ad));
    auto G=block_sparse_bench::propagator(ad,mesh);
    gpu::DiagramEvaluator reference(model,poles,mesh);
    BatchEvaluator candidate(model,poles,mesh);
    exact(candidate.compute_self_energy_range(G,nca,0,1),reference.compute_self_energy_range(G,nca,0,1));
    if (std::string(partition)=="occupation") EXPECT_GT(candidate.last_nca_batch_paths(),0);
    else EXPECT_EQ(candidate.last_nca_batch_paths(),0);
    auto actual=candidate.compute_single_ptcle_gf_range(G,nca,0,1);
    auto expected=reference.compute_single_ptcle_gf_range(G,nca,0,1);
    EXPECT_EQ(nda::max_element(nda::abs(actual-expected)),0.);
    EXPECT_EQ(candidate.last_nca_batch_paths(),0);
    exact(candidate.compute_self_energy_range(G,nca,0,0),reference.compute_self_energy_range(G,nca,0,0));
    EXPECT_EQ(candidate.last_nca_batch_paths(),0);
    auto bound=candidate.self_energy_storage_bound(nca);
    EXPECT_GE(candidate.self_energy_storage_bound(),bound);
    BatchEvaluator limited(model,poles,mesh,bound-1);
    EXPECT_THROW(limited.compute_self_energy_range(G,nca,0,1),std::length_error);
    exact(limited.compute_self_energy_range(G,nca,0,0),reference.compute_self_energy_range(G,nca,0,0));
  }
}
TEST(ProductionRealNca, PolicyAvailability) {
  auto ad=block_sparse_bench::atom(2,"occupation");
  auto model=gpu::export_model(ad,block_sparse_bench::coefficients(ad));
  triqs::mesh::dlr_imtime mesh(2.,triqs::mesh::Fermion,20.,1e-10,false);
  nda::vector<double> poles{-1.5,.6};
  if (!gpu::real_nca_available()) EXPECT_THROW(BatchEvaluator(model,poles,mesh),std::runtime_error);
  EXPECT_THROW(gpu::DiagramEvaluator(model,poles,mesh,1ULL<<30,32ULL<<20,gpu::SparseBackend::Library,
    gpu::SpecialBackend::Reference,gpu::OutputPlacement::Resident,512ULL<<20,static_cast<gpu::NcaBackend>(-1)),std::invalid_argument);
}
#else
#include <gtest/gtest.h>
TEST(ProductionRealNca, DISABLED_NoCuda) {}
#endif

#include <gtest/gtest.h>
#include "diagram_evaluator.hpp"
#include "benchmarks/dense_cuda_small/harness.hpp"
#include <triqs_xca/dense/cuda/diagram_evaluator.hpp>
#include "test_utils/block_sparse.hpp"

TEST(NativeRankChannels, PolesInteractionsAndObservables) {
  using namespace dense_cuda_bench;
  using triqs::operators::n;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using op_t=triqs::operators::many_body_operator_complex;
  for (int lambda : {40,120}) for (int N : {4,8}) for (int pole_count : {2,3}) for (int channels : {0,2}) {
    SCOPED_TRACE(::testing::Message()<<"lambda="<<lambda<<" N="<<N<<" poles="<<pole_count<<" channels="<<channels);
    int nf=std::countr_zero(unsigned(N));
    triqs::atom_diag::fundamental_operator_set fops;
    op_t H;
    for (int i=0;i<nf;++i) { fops.insert("0",i); H+=(.03+.1*i)*n("0",i); }
    H+=.17*n("0",0)*n("0",1)+complex{0.,.05}*c_dag("0",0)*c("0",1)-complex{0.,.05}*c_dag("0",1)*c("0",0);
    triqs::atom_diag::atom_diag<true> ad(H,fops,std::vector<op_t>{});
    ASSERT_EQ(ad.n_subspaces(),1);
    auto G=triqs_xca::test_utils::ad_to_atom_prop(ad,2.,lambda,1e-12);
    auto mesh=G[0].mesh();
    ASSERT_EQ(mesh.dlr_it().rank(),lambda==40?26:36);
    nda::vector<double> poles(pole_count);
    tensor coefficients=nda::zeros<complex>(pole_count,nf,nf);
    tensor interactions=nda::zeros<complex>(pole_count,channels,channels);
    for (int p=0;p<pole_count;++p) {
      poles(p)=-1.5+1.1*p;
      for (int i=0;i<nf;++i) for(int j=0;j<nf;++j)
        coefficients(p,i,j)=i==j?complex{.12+.03*p}:complex{.006*std::cos(i-j),.006*std::sin(i-j)};
      if(channels) {
        interactions(p,0,0)=.012+.002*p;
        interactions(p,1,1)=.017+.001*p;
        interactions(p,0,1)=complex{0.,.003}; interactions(p,1,0)=complex{0.,-.003};
      }
    }
    std::vector<op_t> operators;
    if(channels) operators={op_t{n("0",0)},c_dag("0",0)*c("0",1)+c_dag("0",1)*c("0",0)};
    using Probe=triqs_xca::dense::cuda_small_probe::DiagramEvaluator;
    using Reference=triqs_xca::dense::cuda::DiagramEvaluator;
    auto candidate=channels?std::make_unique<Probe>(poles,coefficients,mesh,ad,operators,interactions):std::make_unique<Probe>(poles,coefficients,mesh,ad);
    auto reference=channels?std::make_unique<Reference>(poles,coefficients,mesh,ad,operators,interactions):std::make_unique<Reference>(poles,coefficients,mesh,ad);
    auto cpu=channels?std::make_unique<triqs_xca::dense::DiagramEvaluator>(poles,coefficients,mesh,ad,operators,interactions):std::make_unique<triqs_xca::dense::DiagramEvaluator>(poles,coefficients,mesh,ad);
    auto low=topology(1,"crossing");
    auto low_indices=select_indices(candidate->get_num_self_energy_backbones(low),32,1729);
    auto gold=reference->compute_self_energy(G,low,low_indices);
    EXPECT_NO_THROW(compare(tensor(gold[0].data()),tensor(cpu->compute_self_energy(G,low,low_indices)[0].data())));
    for (std::string name : {"crossing","chain"}) {
      auto t=topology(3,name);
      for(int count : {8,32,129}) {
        auto indices=select_indices(candidate->get_num_self_energy_backbones(t),count,31415);
        indices(indices.size()-1)=indices(0);
        for(bool pairs : {false,true}) {
          auto expected=pairs?reference->compute_self_energy_by_pairs(G,t,indices):reference->compute_self_energy(G,t,indices);
          auto before=candidate->batch_chunks();
          auto actual=pairs?candidate->compute_self_energy_by_pairs(G,t,indices):candidate->compute_self_energy(G,t,indices);
          EXPECT_NO_THROW(compare(tensor(actual[0].data()),tensor(expected[0].data())));
          EXPECT_GT(candidate->batch_chunks(),before);
        }
      }
      auto indices=select_indices(candidate->get_num_single_ptcle_gf_backbones(t),8,1729);
      EXPECT_NO_THROW(compare(candidate->compute_single_ptcle_gf(G,t,indices),reference->compute_single_ptcle_gf(G,t,indices)));
      std::vector<op_t> tau{op_t{n("0",0)}},zero{op_t{n("0",1)}};
      EXPECT_NO_THROW(compare(candidate->compute_one_time_correlator(G,tau,zero,ad,t,indices),reference->compute_one_time_correlator(G,tau,zero,ad,t,indices)));
      G[0].data()*=complex{.99,.001};
    }
  }
}

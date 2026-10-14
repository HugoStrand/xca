// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <triqs_xca/block_sparse/cuda/real_nca.hpp>
#include <gtest/gtest.h>
namespace gpu=triqs_xca::block_sparse::cuda;
TEST(RealNcaStorage, CheckedPayload) {
#if defined(NDA_HAVE_CUDA) && defined(XCA_HAVE_BLOCK_SPARSE_KERNELS)
  EXPECT_TRUE(gpu::real_nca_available());
#else
  EXPECT_FALSE(gpu::real_nca_available());
#endif
  EXPECT_GT(gpu::real_nca_storage_bound(7,3,2),0);
  EXPECT_THROW(gpu::real_nca_storage_bound(0,3,2),std::invalid_argument);
  EXPECT_THROW(gpu::real_nca_storage_bound(7,3,4),std::invalid_argument);
  EXPECT_THROW(gpu::real_nca_storage_bound(7,std::numeric_limits<long>::max(),1),std::overflow_error);
}
#ifdef NDA_HAVE_CUDA
namespace {
gpu::SparseBlock scalar(double value) {
  gpu::SparseBlock b;b.rows=b.columns=1;
  b.row_offsets={0,1};b.column_indices={0};b.values={value};return b;
}
}
TEST(RealNcaPrimitive, OrderedSpecialCutoffTailAndMetadataRecovery) {
  if(!gpu::real_nca_available()) {
    auto ops=std::make_shared<gpu::DeviceOps>();
    auto table=nda::zeros<nda::dcomplex>(1,1,1);
    EXPECT_THROW(gpu::RealNcaBatch(ops,table,table,1,1ULL<<20),std::runtime_error);
    return;
  }
  constexpr long rank=7,fields_count=3,width=2;
  auto ops=std::make_shared<gpu::DeviceOps>();
  auto forward=nda::zeros<nda::dcomplex>(rank,fields_count,fields_count), reflected=forward;
  for(long t=0;t<rank;++t)for(long i=0;i<fields_count;++i)for(long j=0;j<fields_count;++j){
    forward(t,i,j)=.13+.009*t+.006*(i-j);reflected(t,i,j)=.17-.002*t+.013*(i+2*j);
  }
  auto bound=gpu::real_nca_storage_bound(rank,fields_count,width);
  EXPECT_THROW(gpu::RealNcaBatch(ops,forward,reflected,width,bound-1),std::length_error);
  gpu::RealNcaBatch batch(ops,forward,reflected,width,bound);EXPECT_EQ(batch.storage_bytes(),bound);
  std::vector<gpu::Tensor> G,Sigma,expected;
  for(int b=0;b<2;++b){
    auto input=nda::zeros<nda::dcomplex>(rank,1,1), output=input;
    for(long t=0;t<rank;++t){input(t,0,0)=b?-.2-.004*t:.5+.013*t;output(t,0,0)={.17,.3};}
    G.emplace_back(input);Sigma.emplace_back(output);expected.emplace_back(output);
  }
  ops->wait_for_default_uploads();ops->finish();
  for(long count:{511L,2L}){
    std::vector<gpu::RealNcaLane> lanes;
    std::vector<int> fields,signs;std::vector<double> left,right;
    for(long lane=0;lane<count;++lane){
      int q=lane%3?1:2;
      lanes.push_back({lane%2,(lane/3)%2,long(fields.size()),long(signs.size()),q,int(lane%2),lane%2?-1:1});
      for(int i=0;i<q;++i){fields.push_back((lane+i)%fields_count);left.push_back(i?-.3:.8);right.push_back(lane%7==0?0.:(i?-.25:.4));}
      for(int i=0;i<q;++i)for(int j=0;j<q;++j)signs.push_back((i+j+lane)%2?-1:1);
    }
    std::size_t contributions=0;
    for(auto const &lane:lanes){
      gpu::Tensor out(rank,1,1),mu(rank,1,1),kap(rank,1,1),weighted(rank,1,1),term(rank,1,1);ops->zero(out);
      for(int i=0;i<lane.size;++i){
        ops->zero(mu);
        for(int j=0;j<lane.size;++j){
          auto F=scalar(right[lane.field_offset+j]);
          gpu::DeviceSparseBlock right_op(ops,F,1ULL<<20,gpu::SparseBackend::Library);
          right_op.apply_right(G[lane.input_block],kap);
          nda::vector<nda::dcomplex> host_weights(rank);
          for(long t=0;t<rank;++t){
            auto a=fields[lane.field_offset+i],b=fields[lane.field_offset+j];
            host_weights(t)=double(signs[lane.sign_offset+i*lane.size+j])*(lane.forward?forward(t,a,b):reflected(t,b,a));
          }
          nda::cuvector<nda::dcomplex> weights(host_weights);
          ops->weight_rows(kap,weights,weighted);ops->axpy(1.,weighted,mu);
        }
        auto F=scalar(left[lane.field_offset+i]);
        gpu::DeviceSparseBlock left_op(ops,F,1ULL<<20,gpu::SparseBackend::Library);
        left_op.apply_left(mu,term);ops->axpy(1.,term,out);
      }
      ops->scale(lane.final_sign,out);
      if(ops->any_abs_greater(out,1e-16)){ops->axpy(1.,out,expected[lane.output_block]);++contributions;}
    }
    EXPECT_EQ(batch.evaluate_add(G,Sigma,lanes,fields,left,right,signs),contributions);
    for(int b=0;b<2;++b)EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(Sigma[b])-nda::to_host(expected[b]))),0.);
    auto before=nda::to_host(Sigma[0]);
    auto invalid=lanes;invalid[0].field_offset=-1;
    EXPECT_THROW(batch.evaluate_add(G,Sigma,invalid,fields,left,right,signs),std::invalid_argument);
    EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(Sigma[0])-before)),0.);
    EXPECT_EQ(batch.evaluate_add(G,Sigma,{}, {}, {}, {}, {}),0);
    EXPECT_THROW(batch.evaluate_add(G,G,lanes,fields,left,right,signs),std::invalid_argument);
  }
}
TEST(RealNcaPrimitive, AdjacentThresholdAndLiveSubnormal) {
  if(!gpu::real_nca_available())GTEST_SKIP()<<"Native NCA kernels disabled";
  auto ops=std::make_shared<gpu::DeviceOps>();
  auto table=nda::zeros<nda::dcomplex>(3,1,1);table()=.5;
  gpu::RealNcaBatch batch(ops,table,table,1,gpu::real_nca_storage_bound(3,1,1));
  std::vector<gpu::RealNcaLane> lanes{{0,0,0,0,1,1,1}};
  for(double amplitude:{std::nextafter(2e-16,0.),2e-16,std::nextafter(2e-16,1.)}){
    auto input=nda::zeros<nda::dcomplex>(3,1,1);input()=amplitude;
    std::vector<gpu::Tensor> G{gpu::Tensor(input)},S{gpu::Tensor(input.shape())};
    ops->zero(S[0]);
    auto count=batch.evaluate_add(G,S,lanes,{0},{1.},{1.},{1});
    EXPECT_EQ(count,std::size_t(amplitude*.5>1e-16));
    auto result=nda::to_host(S[0]);
    EXPECT_EQ(result(0,0,0).real(),count?amplitude*.5:0.);
  }
  auto input=nda::ones<nda::dcomplex>(3,1,1);input(0,0,0)=1e-310;
  std::vector<gpu::Tensor> G{gpu::Tensor(input)},S{gpu::Tensor(input.shape())};ops->zero(S[0]);
  EXPECT_EQ(batch.evaluate_add(G,S,lanes,{0},{1.},{1.},{1}),1);
  EXPECT_EQ(nda::to_host(S[0])(0,0,0).real(),.5e-310);
}
#else
TEST(RealNcaPrimitive, DISABLED_RequiresCuda){}
#endif

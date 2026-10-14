// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "fixture.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
namespace gpu=triqs_xca::block_sparse::cuda;
int main(int argc,char **argv) {
  try {
    long M=65,K=64,N=63,B=16,samples=3;double lambda=20.;
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];if(++i==argc)throw std::invalid_argument("Missing value");
      if(key=="--m")M=std::stol(argv[i]);else if(key=="--k")K=std::stol(argv[i]);else if(key=="--n")N=std::stol(argv[i]);
      else if(key=="--rows")B=std::stol(argv[i]);else if(key=="--samples")samples=std::stol(argv[i]);
      else if(key=="--lambda")lambda=std::stod(argv[i]);else throw std::invalid_argument("Unknown option");
    }
    if(M<1||K<1||N<1||M>512||K>512||N>512||B<1||B>512||samples<1||samples>5||!std::isfinite(lambda)||lambda<10||lambda>120)
      throw std::invalid_argument("Beyond bounded row replay");
    cppdlr::imtime_ops host(lambda,cppdlr::build_dlr_rf(lambda,1e-12,false),false);long r=host.rank();
    auto ops=std::make_shared<gpu::DeviceOps>();gpu::RowDlrOps row(ops,host,1ULL<<30);gpu::DlrOps reference(ops,host,1ULL<<30);
    auto f=row_convolution::coefficients(r,M,K,.3),g=row_convolution::coefficients(r,K,N,.8);
    gpu::Tensor fd(f),gd(g),out(r,M,N);reference.convolve(2.,fd,gd,out);auto gold=nda::to_host(out);auto cpu=host.convolve(2.,f,g,cppdlr::TIME_ORDERED);
    auto full=reference.storage_bytes();row.convolve_rows(2.,fd,gd,out,B);
    auto row_gold=nda::to_host(out);
    auto check=[&](bool tiled) {
      auto actual=nda::to_host(out);double error=row_convolution::error(actual,gold),signal=nda::max_element(nda::abs(gold));
      for(auto z:actual)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("Nonfinite row replay");
      if(row_convolution::error(actual,tiled?row_gold:gold)!=0.)throw std::runtime_error("Nondeterministic replay");
      if(error>1e-13*signal+1e-14||row_convolution::error(actual,cpu)>1e-13*nda::max_element(nda::abs(cpu))+1e-14)throw std::runtime_error("Numerical control failed");
      return error;
    };check(true);
    std::cout<<std::setprecision(17)<<"route,rank,m,k,n,rows,sample,call_ms,workspace_bound_bytes,retained_bytes,max_abs_error\n";
    for(long s=0;s<samples;++s)for(bool tiled:{false,true}) {
      auto start=std::chrono::steady_clock::now();
      if(tiled)row.convolve_rows(2.,fd,gd,out,B);else reference.convolve(2.,fd,gd,out);
      double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();auto error=check(tiled);
      std::cout<<(tiled?"rows":"full")<<','<<r<<','<<M<<','<<K<<','<<N<<','<<B<<','<<s<<','<<ms<<','
        <<(tiled?row.row_storage_bound(M,K,N,B):full)<<','<<(tiled?row.storage_bytes():reference.storage_bytes())<<','<<error<<'\n';
    }
  } catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}
}

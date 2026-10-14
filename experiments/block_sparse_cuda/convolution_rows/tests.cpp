// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "fixture.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
namespace gpu=triqs_xca::block_sparse::cuda;
TEST(RowConvolution, RectangularTailsAndScratchRelease) {
  for(double lambda:{10.,40.,120.}) {
    cppdlr::imtime_ops host(lambda,cppdlr::build_dlr_rf(lambda,1e-12,false),false);
    auto ops=std::make_shared<gpu::DeviceOps>();
    gpu::RowDlrOps row(ops,host);gpu::DlrOps reference(ops,host);
    auto tables=row.storage_bytes();long r=host.rank();
    for(auto [M,K,N]:std::vector<std::tuple<long,long,long>>{{1,1,1},{7,5,3},{13,9,11},{4,6,9}}) {
      auto f=row_convolution::coefficients(r,M,K,.3),g=row_convolution::coefficients(r,K,N,.8);
      gpu::Tensor fd(f),gd(g),gold(r,M,N),out(gold.shape()),rhs(f.shape());
      auto cpu=host.convolve(2.,f,g,cppdlr::TIME_ORDERED);
      reference.convolve(2.,fd,gd,gold);auto expected=nda::to_host(gold);
      for(long B:{1L,3L,8L,32L}) {
        row.vals2coefs(fd,rhs); // Retained solve RHS must be released before streaming.
        row.convolve_rows(2.,fd,gd,out,B);auto actual=nda::to_host(out);
        auto tolerance=1e-13*nda::max_element(nda::abs(expected))+1e-14;
        EXPECT_LE(row_convolution::error(actual,expected),tolerance);
        EXPECT_LE(row_convolution::error(actual,cpu),1e-13*nda::max_element(nda::abs(cpu))+1e-14);
        EXPECT_EQ(row.storage_bytes(),tables);
      }
    }
  }
}
TEST(RowConvolution, AdmissionAliasesAndRecovery) {
  cppdlr::imtime_ops host(20.,cppdlr::build_dlr_rf(20.,1e-10,false),false);
  auto ops=std::make_shared<gpu::DeviceOps>();long r=host.rank();
  gpu::RowDlrOps probe(ops,host);auto bound=probe.row_storage_bound(13,9,11,3);
  auto f=row_convolution::coefficients(r,13,9,.3),g=row_convolution::coefficients(r,9,11,.8);
  gpu::Tensor fd(f),gd(g),out(r,13,11);ops->zero(out);ops->finish();
  gpu::RowDlrOps limited(ops,host,bound-1),exact(ops,host,bound);
  EXPECT_THROW(limited.convolve_rows(2.,fd,gd,out,3),std::length_error);
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(out))),0.);
  EXPECT_THROW(exact.convolve_rows(2.,fd,gd,out,0),std::invalid_argument);
  gpu::Tensor square(row_convolution::coefficients(r,9,9,.8));
  EXPECT_THROW(exact.convolve_rows(2.,fd,square,fd,3),std::invalid_argument);
  EXPECT_EQ(row_convolution::error(nda::to_host(fd),f),0.);
  exact.convolve_rows(2.,fd,gd,out,3);
  EXPECT_LE(row_convolution::error(nda::to_host(out),host.convolve(2.,f,g,cppdlr::TIME_ORDERED)),1e-14);
  EXPECT_EQ(probe.row_storage_bound(13,9,11,13),probe.row_storage_bound(13,9,11,100));
  EXPECT_THROW(probe.row_storage_bound(1,1,1,-1),std::invalid_argument);
  EXPECT_THROW(probe.row_storage_bound(std::numeric_limits<long>::max(),std::numeric_limits<long>::max(),1,1),std::overflow_error);
}

TEST(RowConvolution, CutoffNeighborhoodAudit) {
  cppdlr::imtime_ops host(40.,cppdlr::build_dlr_rf(40.,1e-12,false),false);
  auto ops=std::make_shared<gpu::DeviceOps>();long r=host.rank();
  gpu::RowDlrOps row(ops,host);gpu::DlrOps reference(ops,host);
  auto f=row_convolution::coefficients(r,13,9,.3),g=row_convolution::coefficients(r,9,11,.8);
  gpu::Tensor fd(f),gd(g),out(r,13,11);
  reference.convolve(2.,fd,gd,out);auto original=nda::to_host(out);
  double scale=1e-16/nda::max_element(nda::abs(original));
  std::cout<<"cutoff_audit,rows,scale,reference_max,row_max,reference_keep,row_keep,max_error\n";
  for(double factor:{std::nextafter(scale,0.),scale,std::nextafter(scale,2*scale)}) {
    auto scaled=f;scaled*=factor;fd=gpu::Tensor(scaled);
    reference.convolve(2.,fd,gd,out);auto gold=nda::to_host(out);
    for(long B:{1L,3L,8L,32L}) {
      row.convolve_rows(2.,fd,gd,out,B);auto actual=nda::to_host(out);
      double a=nda::max_element(nda::abs(gold)),b=nda::max_element(nda::abs(actual)),error=row_convolution::error(actual,gold);
      EXPECT_LE(error,1e-13*std::max(a,b));
      std::cout<<std::setprecision(17)<<"cutoff_audit,"<<B<<','<<factor<<','<<a<<','<<b<<','<<(a>1e-16)<<','<<(b>1e-16)<<','<<error<<'\n';
    }
  }
}

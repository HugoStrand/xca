// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "row_dlr.hpp"
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cmath>
namespace row_convolution {
inline nda::array<nda::dcomplex,3> coefficients(long r,long m,long n,double phase) {
  nda::array<nda::dcomplex,3> result(r,m,n);
  for(long t=0;t<r;++t)for(long i=0;i<m;++i)for(long j=0;j<n;++j)
    result(t,i,j)={std::sin(.7*t+.3*i-.2*j+phase)/(r*std::max(m,n)),std::cos(.1*t-.4*i+.8*j+phase)/(r*std::max(m,n))};
  return result;
}
inline double error(auto const &a,auto const &b) {return nda::max_element(nda::abs(a-b));}
}

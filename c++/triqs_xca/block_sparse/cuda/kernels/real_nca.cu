// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "kernels.hpp"
#include <cuda_runtime.h>
namespace triqs_xca::block_sparse::cuda::native {
__global__ void nca_values(RealNcaDeviceLane const *lanes,int const *fields,double const *left,double const *right,
                          int const *signs,double const *tables,double *values,long rank,long nf,long count,long capacity) {
  long index=long(blockIdx.x)*blockDim.x+threadIdx.x;
  if (index>=rank*capacity) return;
  long node=index/capacity,lane=index%capacity;
  if (lane>=count) {values[index]=0.;return;}
  auto d=lanes[lane];
  double input=reinterpret_cast<double2 const *>(d.input)[node].x, out=0.;
  for (int i=0;i<d.size;++i) {
    double mu=0.;
    for (int j=0;j<d.size;++j) {
      long a=fields[d.field_offset+i],b=fields[d.field_offset+j];
      long table_index=d.forward ? node*nf*nf+a*nf+b : rank*nf*nf+node*nf*nf+b*nf+a;
      double kap=input*right[d.field_offset+j];
      double weight=double(signs[d.sign_offset+long(i)*d.size+j])*tables[table_index];
      double term=kap*weight;
      mu+=term;
    }
    double term=mu*left[d.field_offset+i];
    out+=term;
  }
  values[index]=out*double(d.final_sign);
}
__global__ void nca_fanin(double const *values,std::uintptr_t const *outputs,long const *offsets,long const *indices,
                          long rank,long capacity,long slots) {
  long index=long(blockIdx.x)*blockDim.x+threadIdx.x;
  if (index>=rank*slots) return;
  long node=index/slots,slot=index%slots;
  auto *output=reinterpret_cast<double2 *>(outputs[slot])+node;
  double sum=output->x;
  for (long k=offsets[slot];k<offsets[slot+1];++k) sum+=values[node*capacity+indices[k]];
  output->x=sum;
}
cudaError_t real_nca_values(RealNcaDeviceLane const *lanes,int const *fields,double const *left,double const *right,int const *signs,
                            double const *tables,double *values,long rank,long nf,long count,long capacity,cudaStream_t stream) {
  nca_values<<<(rank*capacity+255)/256,256,0,stream>>>(lanes,fields,left,right,signs,tables,values,rank,nf,count,capacity);
  return cudaGetLastError();
}
cudaError_t real_nca_fanin(double const *values,std::uintptr_t const *outputs,long const *offsets,long const *indices,
                           long rank,long capacity,long slots,cudaStream_t stream) {
  nca_fanin<<<(rank*slots+255)/256,256,0,stream>>>(values,outputs,offsets,indices,rank,capacity,slots);
  return cudaGetLastError();
}
}

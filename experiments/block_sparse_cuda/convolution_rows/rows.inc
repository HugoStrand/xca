// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#ifdef NDA_HAVE_CUDA
namespace triqs_xca::block_sparse::cuda {
std::size_t RowDlrOps::row_storage_bound(long M, long K, long N, long rows) const {
  if (M<=0 || K<=0 || N<=0 || rows<=0) throw std::invalid_argument("Invalid row convolution dimensions");
  auto B=std::min(rows,M);
  auto bytes=checked_add(tensor_bytes(impl_->r,K,N),checked_multiply(2,tensor_bytes(impl_->r,B,K)));
  bytes=checked_add(bytes,checked_multiply(3,tensor_bytes(impl_->r,B,N)));
  return checked_add(impl_->tables,bytes);
}
void RowDlrOps::convolve_rows(double beta, TensorConstView fc, TensorConstView gc, TensorView out, long rows) {
  auto &x=*impl_;x.validate(fc);x.validate(gc);x.validate(out);
  long M=fc.extent(1),K=fc.extent(2),N=gc.extent(2);
  if (!std::isfinite(beta) || gc.extent(1)!=K || out.shape()!=std::array<long,3>{x.r,M,N})
    throw std::invalid_argument("Invalid row convolution shape");
  auto overlaps=[](TensorConstView a,TensorConstView b) {
    auto p=reinterpret_cast<std::uintptr_t>(a.data()),q=reinterpret_cast<std::uintptr_t>(b.data());
    auto na=checked_multiply(a.size(),sizeof(Complex)),nb=checked_multiply(b.size(),sizeof(Complex));
    return na>UINTPTR_MAX-p || nb>UINTPTR_MAX-q || (p<q+nb && q<p+na);
  };
  if (overlaps(fc,out) || overlaps(gc,out)) throw std::invalid_argument("Row convolution requires disjoint output");
  x.admit(row_storage_bound(M,K,N,rows)); // Before release, allocation or output writes.
  release_scratch();
  Tensor hg(x.r,K,N);
  try {
    x.ops->rank_product(x.H,gc,hg);
    x.ops->finish();
    for (long first=0;first<M;) {
      long B=std::min(rows,M-first);
      Tensor f(x.r,B,K),hf(f.shape()),product(x.r,B,N),cross(product.shape()),diagonal(product.shape());
      try {
        if (observer_) observer_();
        check_cuda(cudaMemcpy2DAsync(f.data(),B*K*sizeof(Complex),fc.data()+first*K,M*K*sizeof(Complex),
                                    B*K*sizeof(Complex),x.r,cudaMemcpyDeviceToDevice,x.ops->stream()));
        x.ops->rank_product(x.H,f,hf);
        x.ops->product(f,gc,product);
        x.ops->rank_product(x.Q,product,diagonal);
        x.ops->product(hf,gc,cross);
        x.ops->product(f,hg,product);
        x.ops->axpy(1.,product,cross);
        x.ops->rank_product(x.C,cross,product);
        x.ops->axpy(1.,product,diagonal);
        x.ops->scale(beta,diagonal);
        check_cuda(cudaMemcpy2DAsync(out.data()+first*N,M*N*sizeof(Complex),diagonal.data(),B*N*sizeof(Complex),
                                    B*N*sizeof(Complex),x.r,cudaMemcpyDeviceToDevice,x.ops->stream()));
        x.ops->finish();
      } catch (...) {try{x.ops->finish();}catch(...){}throw;}
      first+=B;
    }
  } catch (...) {try{x.ops->finish();}catch(...){}throw;}
}
}
#endif

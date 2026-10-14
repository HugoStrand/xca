// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "real_nca.hpp"
#ifdef NDA_HAVE_CUDA
#include "kernels/kernels.hpp"
#include <cmath>
#include <map>
#include <type_traits>
#endif
namespace triqs_xca::block_sparse::cuda {
bool real_nca_available() {
#if defined(NDA_HAVE_CUDA) && defined(XCA_HAVE_BLOCK_SPARSE_KERNELS)
  return true;
#else
  return false;
#endif
}
#ifdef NDA_HAVE_CUDA
namespace {
void check(cudaError_t status) { if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
long extent(std::size_t size) {
  if(size>std::size_t(std::numeric_limits<long>::max())) throw std::overflow_error("Real NCA extent overflow");
  return long(size);
}
struct Context {
  int device=-1,restore=-1;
  Context(){check(cudaGetDevice(&device));}
  void cleanup() noexcept {
    int current;
    if(cudaGetDevice(&current)==cudaSuccess && current!=device && cudaSetDevice(device)==cudaSuccess) restore=current;
  }
  ~Context(){if(restore>=0)cudaSetDevice(restore);}
};
}
struct RealNcaBatch::Impl {
  Context context;
  std::shared_ptr<DeviceOps> ops;
  long rank,fields,width;
  std::size_t bound;
  nda::cuvector<double> tables,left,right;
  nda::cuvector<int> indices,signs;
  nda::cuvector<RealNcaDeviceLane> lanes;
  nda::cuarray<double,2> values;
  nda::cuvector<std::uintptr_t> destinations;
  nda::cuvector<long> offsets,selected;
  Impl(std::shared_ptr<DeviceOps> owner,nda::array_const_view<nda::dcomplex,3> forward,
       nda::array_const_view<nda::dcomplex,3> reflected,long max_width,std::size_t budget)
    :ops(std::move(owner)),rank(forward.extent(0)),fields(forward.extent(1)),width(max_width),bound(0) {
    if(!real_nca_available())throw std::runtime_error("Real NCA kernels were not built");
    if(!ops)throw std::invalid_argument("Missing real NCA owner");
    ops->require_current();
    if(forward.shape()!=reflected.shape() || forward.extent(2)!=fields)throw std::invalid_argument("Real NCA table shape mismatch");
    bound=real_nca_storage_bound(rank,fields,width);
    if(bound>budget)throw std::length_error("Real NCA storage exceeds budget");
    auto total=checked_multiply(rank,real_nca_capacity);
    if(checked_add(total,255)/256>std::size_t(std::numeric_limits<int>::max()))throw std::length_error("Real NCA grid limit");
    nda::vector<double> host(extent(checked_multiply(2,forward.size())));
    for(long t=0;t<rank;++t)for(long i=0;i<fields;++i)for(long j=0;j<fields;++j) {
      auto a=forward(t,i,j),b=reflected(t,i,j);
      if(!std::isfinite(a.real())||!std::isfinite(b.real())||a.imag()!=0.||b.imag()!=0.)throw std::invalid_argument("Real NCA requires finite real tables");
      auto index=(t*fields+i)*fields+j;host(index)=a.real();host(forward.size()+index)=b.real();
    }
    auto slots=extent(checked_multiply(real_nca_capacity,width));
    tables=nda::cuvector<double>(host.size());left=nda::cuvector<double>(slots);right=nda::cuvector<double>(slots);
    indices=nda::cuvector<int>(slots);signs=nda::cuvector<int>(extent(checked_multiply(slots,width)));
    static_assert(std::is_trivially_default_constructible_v<RealNcaDeviceLane>);
    lanes=nda::cuvector<RealNcaDeviceLane>(real_nca_capacity);values=nda::cuarray<double,2>(rank,real_nca_capacity);
    destinations=nda::cuvector<std::uintptr_t>(real_nca_capacity);offsets=nda::cuvector<long>(real_nca_capacity+1);selected=nda::cuvector<long>(real_nca_capacity);
    try {check(cudaMemcpyAsync(tables.data(),host.data(),host.size()*sizeof(double),cudaMemcpyHostToDevice,ops->stream()));ops->finish();}
    catch(...){try{ops->finish();}catch(...){}throw;}
  }
  ~Impl(){context.cleanup();if(ops)try{ops->finish();}catch(...) {}}
  template<typename T>void upload(nda::cuvector<T>&target,std::vector<T>const&source){
    if(!source.empty())check(cudaMemcpyAsync(target.data(),source.data(),source.size()*sizeof(T),cudaMemcpyHostToDevice,ops->stream()));
  }
};
RealNcaBatch::RealNcaBatch(std::shared_ptr<DeviceOps> ops,nda::array_const_view<nda::dcomplex,3> forward,
                          nda::array_const_view<nda::dcomplex,3> reflected,long width,std::size_t budget)
  :impl_(std::make_unique<Impl>(std::move(ops),forward,reflected,width,budget)){}
RealNcaBatch::~RealNcaBatch()=default;
std::size_t RealNcaBatch::storage_bytes()const{return impl_->bound;}
std::size_t RealNcaBatch::evaluate_add(std::vector<Tensor>const&G,std::vector<Tensor>&Sigma,
    std::vector<RealNcaLane>const&lanes,std::vector<int>const&fields,std::vector<double>const&left,
    std::vector<double>const&right,std::vector<int>const&signs) {
  std::vector<TensorConstView> input;
  std::vector<TensorView> output;
  input.reserve(G.size()); output.reserve(Sigma.size());
  for(auto const &tensor:G) input.emplace_back(tensor);
  for(auto &tensor:Sigma) output.emplace_back(tensor);
  return evaluate_add_views(input,output,lanes,fields,left,right,signs);
}
std::size_t RealNcaBatch::evaluate_add_views(std::span<TensorConstView const> G,std::span<TensorView const> Sigma,
    std::vector<RealNcaLane>const&lanes,std::vector<int>const&fields,std::vector<double>const&left,
    std::vector<double>const&right,std::vector<int>const&signs) {
#ifndef XCA_HAVE_BLOCK_SPARSE_KERNELS
  throw std::runtime_error("Real NCA kernels were not built");
#else
  auto &x=*impl_;x.ops->require_current();
  if(lanes.size()>real_nca_capacity || fields.size()>std::size_t(x.indices.size()) || signs.size()>std::size_t(x.signs.size())
      ||left.size()!=fields.size()||right.size()!=fields.size())throw std::invalid_argument("Real NCA metadata size mismatch");
  for(int f:fields)if(f<0||f>=x.fields)throw std::invalid_argument("Real NCA field index");
  for(double v:left)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite real NCA coefficient");
  for(double v:right)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite real NCA coefficient");
  for(int s:signs)if(s!=-1&&s!=1)throw std::invalid_argument("Invalid real NCA parity");
  std::vector<RealNcaDeviceLane> metadata;metadata.reserve(lanes.size());
  for(auto const &lane:lanes) {
    if(lane.input_block<0||lane.input_block>=long(G.size())||lane.output_block<0||lane.output_block>=long(Sigma.size())
       ||lane.size<=0||lane.size>x.width||lane.field_offset<0||lane.sign_offset<0||lane.forward<0||lane.forward>1
       ||(lane.final_sign!=-1&&lane.final_sign!=1))throw std::invalid_argument("Invalid real NCA lane");
    if(checked_add(lane.field_offset,lane.size)>fields.size()
       ||checked_add(lane.sign_offset,checked_multiply(lane.size,lane.size))>signs.size())throw std::invalid_argument("Real NCA lane offsets");
    auto shape=std::array<long,3>{x.rank,1,1};
    if(G[lane.input_block].shape()!=shape||Sigma[lane.output_block].shape()!=shape
       || !G[lane.input_block].indexmap().is_contiguous() || !Sigma[lane.output_block].indexmap().is_contiguous())throw std::invalid_argument("Real NCA tensor shape");
    metadata.push_back({reinterpret_cast<std::uintptr_t>(G[lane.input_block].data()),lane.field_offset,lane.sign_offset,
                        lane.size,lane.forward,lane.final_sign});
  }
  auto bytes=checked_multiply(x.rank,sizeof(nda::dcomplex));
  for(auto const &input:metadata)for(auto const &lane:lanes) {
    auto output=reinterpret_cast<std::uintptr_t>(Sigma[lane.output_block].data());
    if(input.input<=output ? output-input.input<bytes : input.input-output<bytes)
      throw std::invalid_argument("Real NCA input/output alias");
  }
  for(auto const &a:lanes)for(auto const &b:lanes) if(a.output_block<b.output_block) {
    auto first=reinterpret_cast<std::uintptr_t>(Sigma[a.output_block].data());
    auto second=reinterpret_cast<std::uintptr_t>(Sigma[b.output_block].data());
    if(first<=second ? second-first<bytes : first-second<bytes)
      throw std::invalid_argument("Real NCA output alias");
  }
  if(lanes.empty())return 0;
  // These host buffers outlive copies/readers even if a later launch fails.
  std::vector<std::uintptr_t> destinations;
  std::vector<long> offsets{0},selected;
  try {
    x.upload(x.lanes,metadata);x.upload(x.indices,fields);x.upload(x.left,left);x.upload(x.right,right);x.upload(x.signs,signs);
    check(native::real_nca_values(x.lanes.data(),x.indices.data(),x.left.data(),x.right.data(),x.signs.data(),x.tables.data(),x.values.data(),
                                 x.rank,x.fields,lanes.size(),real_nca_capacity,x.ops->stream()));
    x.ops->finish();auto host=nda::to_host(x.values);
    std::map<long,std::vector<long>> groups;
    for(long lane=0;lane<long(lanes.size());++lane){
      double magnitude=0;
      for(long t=0;t<x.rank;++t){auto v=host(t,lane);if(!std::isfinite(v))throw std::runtime_error("Nonfinite real NCA completed value");magnitude=std::max(magnitude,std::abs(v));}
      if(magnitude>1e-16)groups[lanes[lane].output_block].push_back(lane);
    }
    for(auto const &[output,sequence]:groups){
      destinations.push_back(reinterpret_cast<std::uintptr_t>(Sigma[output].data()));
      selected.insert(selected.end(),sequence.begin(),sequence.end());offsets.push_back(selected.size());
    }
    if(!groups.empty()){
      x.upload(x.destinations,destinations);x.upload(x.offsets,offsets);x.upload(x.selected,selected);
      check(native::real_nca_fanin(x.values.data(),x.destinations.data(),x.offsets.data(),x.selected.data(),x.rank,real_nca_capacity,groups.size(),x.ops->stream()));
      x.ops->finish();
    }
    return selected.size();
  }catch(...){try{x.ops->finish();}catch(...){}throw;}
#endif
}
#endif
}

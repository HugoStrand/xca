// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "device_ops.hpp"
#include "memory.hpp"
#include "real_nca_types.hpp"
namespace triqs_xca::block_sparse::cuda {
bool real_nca_available();
inline constexpr long real_nca_capacity = 512;
inline std::size_t real_nca_storage_bound(long rank,long fields,long width) {
  if (rank<=0 || fields<=0 || width<=0 || width>fields) throw std::invalid_argument("Invalid real NCA dimensions");
  auto slots=checked_multiply(real_nca_capacity,width);
  auto tables=checked_multiply(2,checked_multiply(sizeof(double),checked_multiply(rank,checked_multiply(fields,fields))));
  auto coefficients=checked_multiply(slots,sizeof(int)+2*sizeof(double));
  auto signs=checked_multiply(checked_multiply(slots,width),sizeof(int));
  auto values=checked_multiply(checked_multiply(rank,real_nca_capacity),sizeof(double));
  auto metadata=checked_multiply(real_nca_capacity,sizeof(RealNcaDeviceLane)+sizeof(std::uintptr_t)+2*sizeof(long));
  return checked_add(tables,checked_add(coefficients,checked_add(signs,checked_add(values,checked_add(metadata,sizeof(long))))));
}
#ifdef NDA_HAVE_CUDA
// Internal synchronous primitive. Caller supplies verified real inputs and a
// finite-intermediate/whole-sum range guard; inputs and outputs are disjoint.
// Outputs may contain prior additions.
// Table/metadata indices and tensor shapes are checked before device submission.
class RealNcaBatch {
  struct Impl;
  std::unique_ptr<Impl> impl_;
public:
  RealNcaBatch(std::shared_ptr<DeviceOps> ops,nda::array_const_view<nda::dcomplex,3> forward,
               nda::array_const_view<nda::dcomplex,3> reflected,long width,std::size_t budget);
  ~RealNcaBatch();
  RealNcaBatch(RealNcaBatch const&)=delete;
  RealNcaBatch& operator=(RealNcaBatch const&)=delete;
  std::size_t storage_bytes() const;
  std::size_t evaluate_add(std::vector<Tensor> const &G,std::vector<Tensor> &Sigma,
    std::vector<RealNcaLane> const &lanes,std::vector<int> const &fields,
    std::vector<double> const &left,std::vector<double> const &right,std::vector<int> const &signs);
};
#endif
}

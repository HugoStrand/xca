// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include "../real_nca_types.hpp"
namespace triqs_xca::block_sparse::cuda::native {
  cudaError_t mapped_weighted_mu(void const *input, long const *sources, void const *values, void const *weights, void *output, long rank, long rows,
                                 long input_columns, long columns, long fields, cudaStream_t stream);
  // CSR rows own outputs; right applies the plain-transposed CSR snapshot.
  cudaError_t ordered_apply(long const *offsets, long const *columns, void const *values, void const *input, void *output, long rank, long rows,
                            long cols, long width, bool right, cudaStream_t stream);
  cudaError_t real_nca_values(RealNcaDeviceLane const *,int const *,double const *,double const *,int const *,double const *,double *,
                              long,long,long,long,cudaStream_t);
  cudaError_t real_nca_fanin(double const *,std::uintptr_t const *,long const *,long const *,long,long,long,cudaStream_t);
} // namespace triqs_xca::block_sparse::cuda::native

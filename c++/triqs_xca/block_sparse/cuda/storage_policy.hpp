// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace triqs_xca::block_sparse::cuda {
  // All policies return complete observables. HostInputs borrows input blocks;
  // StreamFragments additionally recomputes correlator fragments in two slots.
  enum class StoragePolicy { Resident, HostInputs, StreamFragments };
  inline bool valid_storage_policy(StoragePolicy policy) noexcept {
    return policy == StoragePolicy::Resident || policy == StoragePolicy::HostInputs || policy == StoragePolicy::StreamFragments;
  }
} // namespace triqs_xca::block_sparse::cuda

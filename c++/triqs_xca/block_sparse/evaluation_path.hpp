// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <vector>

namespace triqs_xca::block_sparse {
  struct SelfEnergyPath {
    long flat_index;
    int output_block, final_block, zero_set, special_vertex;
    std::vector<int> propagator_blocks; // 2*m-1 internal edges, in traversal order.
    std::vector<long> dimensions;       // Input/output dimension at each of 2*m vertices.
    bool operator==(SelfEnergyPath const &) const = default;
  };
} // namespace triqs_xca::block_sparse

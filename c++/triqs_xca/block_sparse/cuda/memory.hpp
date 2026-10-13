// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nda/nda.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

namespace triqs_xca::block_sparse::cuda {
  inline std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::overflow_error("Block-sparse memory sum overflow");
    return a + b;
  }
  inline std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) throw std::overflow_error("Block-sparse memory product overflow");
    return a * b;
  }
  inline std::size_t checked_extent(long value) {
    if (value < 0) throw std::invalid_argument("Negative block-sparse extent");
    return static_cast<std::size_t>(value);
  }
  inline std::size_t tensor_bytes(long rank, long rows, long columns) {
    return checked_multiply(sizeof(nda::dcomplex),
                            checked_multiply(checked_extent(rank), checked_multiply(checked_extent(rows), checked_extent(columns))));
  }
  inline std::size_t function_bytes(long rank, std::vector<long> const &dimensions) {
    std::size_t bytes = 0;
    for (long dim : dimensions) bytes = checked_add(bytes, tensor_bytes(rank, dim, dim));
    return bytes;
  }
  struct MemoryInventory {
    std::size_t function = 0, maximum_rank_buffer = 0, legacy_scratch = 0, legacy_correlator_sides = 0;
  };
  // Capacity diagnostics for the retained CPU layout, not admission for a CUDA
  // implementation. Actual device admission must include its declared live
  // buffers, model, output, library workspace, caches and free-memory headroom.
  inline MemoryInventory inventory_memory(long rank, std::vector<long> const &dimensions, long maximum_set_size) {
    if (maximum_set_size < 0) throw std::invalid_argument("Negative symmetry-set size");
    long largest = 0;
    for (long dim : dimensions) {
      (void)checked_extent(dim);
      if (dim > largest) largest = dim;
    }
    auto tile = tensor_bytes(rank, largest, largest);
    return {function_bytes(rank, dimensions), tile, checked_multiply(checked_add(4, checked_extent(maximum_set_size)), tile),
            checked_multiply(checked_multiply(2, dimensions.size()), tile)};
  }
} // namespace triqs_xca::block_sparse::cuda

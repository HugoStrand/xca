// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nda/nda.hpp>
#include <algorithm>
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

  inline std::size_t reference_storage_bound(long rank, std::vector<long> const &dimensions, std::size_t dlr_bound,
                                             std::size_t operator_budget, std::size_t kernel_bytes, std::size_t mapped_bytes,
                                             std::size_t rank_buffers = 7) {
    auto tile = inventory_memory(rank, dimensions, 0).maximum_rank_buffer;
    if (tile / sizeof(nda::dcomplex) > std::size_t(std::numeric_limits<int>::max()))
      throw std::length_error("CUDA reference rank buffer exceeds BLAS vector limit");
    return checked_add(checked_add(checked_multiply(2, function_bytes(rank, dimensions)), checked_multiply(rank_buffers, tile)),
                       checked_add(dlr_bound, checked_add(operator_budget,
                         checked_add(checked_add(kernel_bytes, mapped_bytes), tensor_bytes(1, rank, 1)))));
  }

  struct ReferenceMemoryPlan {
    std::size_t device_nca, device_convolution, device_correlator, host_output;
    std::size_t function, maximum_rank_buffer, fixed_tables, mapped_metadata;
  };
  // Host-only preflight of the production reference's explicit array payload.
  // mapped_set_width is zero for Reference, or the largest non-singleton set for
  // Mapped. Library allocations, CUDA context/headroom, host model construction,
  // and experimental scalar buffers are deliberately outside these bounds.
  inline ReferenceMemoryPlan reference_memory_plan(long rank, std::vector<long> const &dimensions, long poles,
                                                    std::size_t operator_budget, long mapped_set_width = 0,
                                                    bool host_sectors = false, long mu = 0, long kap = 0) {
    if (rank <= 0 || dimensions.empty() || poles <= 0 || operator_budget == 0)
      throw std::invalid_argument("Invalid CUDA reference memory plan");
    for (long dim : dimensions)
      if (dim <= 0) throw std::invalid_argument("Invalid CUDA reference block dimension");
    (void)checked_extent(mapped_set_width);
    (void)checked_extent(mu);
    (void)checked_extent(kap);
    auto inventory = inventory_memory(rank, dimensions, 0);
    auto tables = checked_add(checked_multiply(5, tensor_bytes(1, rank, rank)),
                              checked_multiply(checked_add(rank, 1), sizeof(int)));
    auto kernels = checked_multiply(2, tensor_bytes(1, rank, poles));
    long maximum = 0;
    for (long dim : dimensions) maximum = std::max(maximum, dim);
    auto mapped = mapped_set_width <= 1 ? 0 : checked_multiply(mapped_set_width,
      checked_add(checked_multiply(maximum, sizeof(long) + sizeof(nda::dcomplex)), tensor_bytes(1, rank, 1)));
    auto dlr = checked_add(tables, checked_multiply(5, inventory.maximum_rank_buffer));
    auto generic = reference_storage_bound(rank, dimensions, dlr, operator_budget, kernels, mapped);
    auto self_energy = host_sectors ? checked_add(generic - inventory.function, inventory.maximum_rank_buffer) : generic;
    auto result = tensor_bytes(rank, mu, kap);
    if (result / sizeof(nda::dcomplex) > std::size_t(std::numeric_limits<int>::max()))
      throw std::length_error("Correlator exceeds BLAS vector range");
    auto correlator = checked_add(generic, checked_add(inventory.function,
      checked_add(checked_multiply(2, result), checked_multiply(maximum, sizeof(nda::dcomplex)))));
    return {self_energy - (dlr - tables) - inventory.maximum_rank_buffer, self_energy, correlator, inventory.function,
            inventory.function, inventory.maximum_rank_buffer, checked_add(tables, kernels), mapped};
  }
} // namespace triqs_xca::block_sparse::cuda

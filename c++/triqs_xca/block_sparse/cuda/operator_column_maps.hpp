// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <triqs_xca/block_sparse/cuda/sparse_operator.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <optional>
namespace triqs_xca::block_sparse::cuda {
  using Complex = nda::dcomplex;
  struct ColumnMaps {
    nda::array<long, 2> sources;
    nda::array<Complex, 2> values;
  };
  inline std::optional<ColumnMaps> make_operator_column_maps(std::vector<triqs_xca::block_sparse::cuda::SparseBlock const *> const &blocks) {
    if (blocks.empty() || !blocks.front()) return std::nullopt;
    auto rows = blocks.front()->rows, columns = blocks.front()->columns;
    if (rows <= 0 || columns <= 0) return std::nullopt;
    auto elements = triqs_xca::block_sparse::cuda::checked_multiply(blocks.size(), columns);
    if (elements > std::size_t(std::numeric_limits<long>::max())) throw std::overflow_error("Column map extent overflow");
    (void)triqs_xca::block_sparse::cuda::checked_multiply(elements, sizeof(long) + sizeof(Complex));
    ColumnMaps result{nda::array<long, 2>(blocks.size(), columns), nda::array<Complex, 2>(blocks.size(), columns)};
    result.sources() = -1;
    result.values()  = 0.;
    for (std::size_t field = 0; field < blocks.size(); ++field) {
      if (!blocks[field]) return std::nullopt;
      auto const &block = *blocks[field];
      block.validate();
      if (block.rows != rows || block.columns != columns) return std::nullopt;
      for (long row = 0; row < rows; ++row)
        for (long k = block.row_offsets[row]; k < block.row_offsets[row + 1]; ++k) {
          long column = block.column_indices[k];
          if (result.sources(field, column) != -1) return std::nullopt;
          result.sources(field, column) = row;
          result.values(field, column)  = block.values[k];
        }
    }
    return result;
  }
} // namespace triqs_xca::block_sparse::cuda

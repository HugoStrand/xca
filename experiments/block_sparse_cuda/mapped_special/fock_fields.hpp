// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "column_maps.hpp"
#include <bit>
#include <map>
namespace mapped_special_experiment {
  // Exact canonical annihilation maps between adjacent number sectors. No AD.
  inline std::vector<triqs_xca::block_sparse::cuda::SparseBlock> fock_fields(int modes, bool creation) {
    if (modes < 2 || modes > 14 || modes % 2) throw std::invalid_argument("Bounded even Fock mode count required");
    std::vector<unsigned> lower, upper;
    for (unsigned state = 0; state < (1u << modes); ++state) {
      if (std::popcount(state) == modes / 2 - 1) lower.push_back(state);
      if (std::popcount(state) == modes / 2) upper.push_back(state);
    }
    std::map<unsigned, long> inverse;
    for (long row = 0; row < long(lower.size()); ++row) inverse[lower[row]] = row;
    std::vector<triqs_xca::block_sparse::cuda::SparseBlock> fields;
    for (int field = 0; field < modes; ++field) {
      std::vector<std::vector<std::pair<long, nda::dcomplex>>> entries(lower.size());
      unsigned bit = 1u << field;
      for (long column = 0; column < long(upper.size()); ++column)
        if (upper[column] & bit) {
          double sign = std::popcount(upper[column] & (bit - 1)) % 2 ? -1. : 1.;
          entries[inverse.at(upper[column] ^ bit)].push_back({column, sign});
        }
      triqs_xca::block_sparse::cuda::SparseBlock block;
      block.rows    = lower.size();
      block.columns = upper.size();
      for (auto const &row : entries) {
        for (auto const &[column, value] : row) {
          block.column_indices.push_back(column);
          block.values.push_back(value);
        }
        block.row_offsets.push_back(block.values.size());
      }
      block.validate();
      fields.push_back(creation ? block.adjoint() : std::move(block));
    }
    return fields;
  }
} // namespace mapped_special_experiment

// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <triqs_xca/atom_diag.hpp>

namespace triqs_xca::block_sparse::cuda {
  // Host model metadata, usable and testable in CPU-only builds. CSR entries are
  // coalesced, column-sorted within each row, and never numerically thresholded.
  struct SparseBlock {
    long rows = 0, columns = 0;
    std::vector<long> row_offsets{0}, column_indices;
    std::vector<nda::dcomplex> values;
    void validate() const;
    nda::matrix<nda::dcomplex> to_dense() const;
    SparseBlock adjoint() const;
    nda::array<nda::dcomplex, 3> apply_left(nda::array_const_view<nda::dcomplex, 3> input) const;
    nda::array<nda::dcomplex, 3> apply_right(nda::array_const_view<nda::dcomplex, 3> input) const;
  };

  struct SparseOperator {
    std::vector<long> dimensions;
    std::vector<int> connection;
    std::vector<SparseBlock> blocks; // Source-subspace order; absent blocks have zero rows.
    // The target survives exact cancellation of entries, matching atom_diag's
    // monomial-path connection policy rather than inferring connectivity from nnz.
    void validate() const;
    SparseOperator adjoint() const;
    std::size_t nonzeros() const;
  };

  SparseOperator export_operator(triqs_xca::atom_diag::triqs_atom_diag_t<true> const &ad, triqs::operators::many_body_operator_complex const &op);
  SparseOperator export_operator(triqs_xca::atom_diag::triqs_atom_diag_t<false> const &ad, triqs::operators::many_body_operator_complex const &op);
  // Preserve the supported dynamic-operator domain: nonzero structural map and
  // at most one source for each target, so its adjoint is also block resolvable.
  void validate_interaction_operator(SparseOperator const &op);
} // namespace triqs_xca::block_sparse::cuda

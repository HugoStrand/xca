// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sparse_operator.hpp"

namespace triqs_xca::block_sparse::cuda {
  // Owned immutable model snapshot. Symmetry sets deliberately retain the CPU
  // grouping policy; more permissive grouping is a separate qualification step.
  class OperatorModel {
    std::vector<SparseOperator> fields_, adjoints_, dagger_bars_, reflected_bars_;
    std::vector<std::vector<int>> sets_;
    std::vector<int> labels_, within_set_;
    nda::array<nda::dcomplex, 3> coefficients_;
    int n_hyb_;
    long poles_;
    std::size_t storage_bound_ = 0;

    public:
    OperatorModel(std::vector<SparseOperator> fields, int n_hyb, nda::array_const_view<nda::dcomplex, 3> coefficients,
                  std::size_t storage_budget = 512ULL * 1024 * 1024);
    int num_fields() const { return fields_.size(); }
    int num_hybridization_fields() const { return n_hyb_; }
    int num_interactions() const { return num_fields() - n_hyb_; }
    long num_poles() const { return poles_; }
    std::vector<long> const &dimensions() const { return fields_[0].dimensions; }
    std::vector<std::vector<int>> const &sets() const { return sets_; }
    std::vector<int> const &labels() const { return labels_; }
    std::vector<int> const &within_set() const { return within_set_; }
    nda::array_const_view<nda::dcomplex, 3> coefficients() const { return coefficients_; }
    SparseOperator const &field(int index, bool dagger = false) const;
    SparseOperator const &barred(int index, long pole, bool dagger) const;
    // Includes owned objects, vector capacities and coefficient payload, but
    // excludes allocator bookkeeping, construction temporaries and atom_diag.
    std::size_t host_storage_bytes() const;
    std::size_t host_storage_bound_bytes() const { return storage_bound_; }
    // CSR array payload only, before device descriptor tables/packing/alignment.
    std::size_t sparse_payload_bytes() const;
  };

  OperatorModel export_model(triqs_xca::atom_diag::triqs_atom_diag_t<true> const &ad, nda::array_const_view<nda::dcomplex, 3> coefficients,
                             std::size_t storage_budget = 512ULL * 1024 * 1024);
  OperatorModel export_model(triqs_xca::atom_diag::triqs_atom_diag_t<false> const &ad, nda::array_const_view<nda::dcomplex, 3> coefficients,
                             std::size_t storage_budget = 512ULL * 1024 * 1024);
  OperatorModel export_model(triqs_xca::atom_diag::triqs_atom_diag_t<true> const &ad, nda::array_const_view<nda::dcomplex, 3> coefficients,
                             std::vector<triqs::operators::many_body_operator_complex> const &interactions,
                             nda::array_const_view<nda::dcomplex, 3> interaction_coefficients, std::size_t storage_budget = 512ULL * 1024 * 1024);
  OperatorModel export_model(triqs_xca::atom_diag::triqs_atom_diag_t<false> const &ad, nda::array_const_view<nda::dcomplex, 3> coefficients,
                             std::vector<triqs::operators::many_body_operator_complex> const &interactions,
                             nda::array_const_view<nda::dcomplex, 3> interaction_coefficients, std::size_t storage_budget = 512ULL * 1024 * 1024);
} // namespace triqs_xca::block_sparse::cuda

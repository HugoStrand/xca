// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "operator_model.hpp"
#include "memory.hpp"
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/hyb.hpp>
#include <algorithm>
#include <map>

namespace triqs_xca::block_sparse::cuda {
  namespace {
    using complex = nda::dcomplex;
    template <typename T> std::size_t vector_bytes(std::vector<T> const &v) { return checked_multiply(v.capacity(), sizeof(T)); }
    std::size_t operator_storage(SparseOperator const &op) {
      auto bytes = checked_add(vector_bytes(op.dimensions), checked_add(vector_bytes(op.connection), vector_bytes(op.blocks)));
      for (auto const &b : op.blocks)
        bytes = checked_add(bytes, checked_add(vector_bytes(b.row_offsets), checked_add(vector_bytes(b.column_indices), vector_bytes(b.values))));
      return bytes;
    }
    std::size_t operator_payload(SparseOperator const &op) {
      std::size_t bytes = 0;
      for (auto const &b : op.blocks)
        bytes = checked_add(bytes,
                            checked_add(checked_multiply(checked_add(b.row_offsets.size(), b.column_indices.size()), sizeof(long)),
                                        checked_multiply(b.values.size(), sizeof(complex))));
      return bytes;
    }
    std::size_t metadata_storage(SparseOperator const &op) {
      return checked_multiply(op.dimensions.size(), sizeof(long) + sizeof(int) + sizeof(SparseBlock));
    }
    std::size_t transpose_storage(SparseOperator const &op) {
      auto bytes = metadata_storage(op);
      for (auto const &b : op.blocks) {
        bytes = checked_add(bytes, sizeof(long)); // An absent block still owns {0}.
        if (b.rows == 0) continue;
        bytes = checked_add(bytes, checked_multiply(checked_extent(b.columns), sizeof(long)));
        bytes = checked_add(bytes, checked_multiply(b.values.size(), sizeof(long) + sizeof(complex)));
      }
      return bytes;
    }
    // Conservative bound for materialized CSR with worst-case no overlap.
    // reserve() uses this same nnz count so the ownership estimate covers capacity.
    SparseOperator combine(std::vector<SparseOperator const *> const &ops, std::vector<complex> const &weights) {
      auto const &rep = *ops[0];
      SparseOperator out{rep.dimensions, rep.connection, std::vector<SparseBlock>(rep.blocks.size())};
      for (std::size_t b = 0; b < out.blocks.size(); ++b) {
        auto &block       = out.blocks[b];
        auto const &shape = rep.blocks[b];
        block.rows        = shape.rows;
        block.columns     = shape.columns;
        if (rep.connection[b] == -1) continue;
        std::size_t upper = 0;
        for (std::size_t k = 0; k < ops.size(); ++k)
          if (weights[k] != complex{}) upper = checked_add(upper, ops[k]->blocks[b].values.size());
        block.values.reserve(upper);
        block.column_indices.reserve(upper);
        block.row_offsets.assign(block.rows + 1, 0);
        for (long row = 0; row < block.rows; ++row) {
          std::map<long, complex> entries;
          for (std::size_t k = 0; k < ops.size(); ++k) {
            if (weights[k] == complex{}) continue;
            auto const &source = ops[k]->blocks[b];
            for (long j = source.row_offsets[row]; j < source.row_offsets[row + 1]; ++j)
              entries[source.column_indices[j]] += weights[k] * source.values[j];
          }
          for (auto const &[column, value] : entries) {
            if (value == complex{}) continue;
            block.column_indices.push_back(column);
            block.values.push_back(value);
          }
          if (block.values.size() > static_cast<std::size_t>(std::numeric_limits<long>::max()))
            throw std::overflow_error("Barred CSR index overflow");
          block.row_offsets[row + 1] = block.values.size();
        }
      }
      out.validate();
      return out;
    }
    template <bool C>
    OperatorModel export_impl(triqs_xca::atom_diag::triqs_atom_diag_t<C> const &ad, nda::array_const_view<complex, 3> coeff,
                              std::vector<triqs::operators::many_body_operator_complex> const &interactions, nda::array_const_view<complex, 3> dyn,
                              std::size_t budget) {
      int n = ad.get_fops().size();
      if (n <= 0 || coeff.extent(0) <= 0 || coeff.extent(1) != n || coeff.extent(2) != n || dyn.extent(0) != coeff.extent(0)
          || dyn.extent(1) != static_cast<long>(interactions.size()) || dyn.extent(2) != static_cast<long>(interactions.size()))
        throw std::invalid_argument("Sparse model coefficient/channel shape mismatch");
      std::vector<SparseOperator> fields;
      fields.reserve(checked_add(n, interactions.size()));
      for (auto const &mode : ad.get_fops())
        fields.push_back(export_operator(ad, triqs::operators::many_body_operator_complex::make_canonical(false, mode.index)));
      for (auto const &op : interactions) fields.push_back(export_operator(ad, op));
      return OperatorModel(std::move(fields), n, hyb::get_extended_coefficients(coeff, dyn), budget);
    }
  } // namespace

  OperatorModel::OperatorModel(std::vector<SparseOperator> fields, int n_hyb, nda::array_const_view<complex, 3> coeff, std::size_t budget)
     : fields_(std::move(fields)), n_hyb_(n_hyb), poles_(coeff.extent(0)) {
    if (fields_.empty() || fields_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || n_hyb <= 0 || n_hyb > num_fields()
        || poles_ <= 0 || coeff.extent(1) != num_fields() || coeff.extent(2) != num_fields())
      throw std::invalid_argument("Invalid sparse model field/pole/coefficient extents");
    auto bars = checked_multiply(checked_extent(poles_), fields_.size());
    auto coefficient_bytes =
       checked_multiply(sizeof(complex), checked_multiply(checked_extent(poles_), checked_multiply(fields_.size(), fields_.size())));
    auto minimum = checked_add(
       sizeof(*this),
       checked_add(coefficient_bytes,
                   checked_multiply(checked_add(checked_multiply(2, fields_.size()), checked_multiply(2, bars)), sizeof(SparseOperator))));
    if (minimum > budget) throw std::length_error("Sparse operator model exceeds owned storage budget");
    auto const &dims = fields_[0].dimensions;
    if (dims.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Sparse subspace count exceeds integer indexing");
    if (dims.empty() || std::any_of(dims.begin(), dims.end(), [](long d) { return d <= 0; }))
      throw std::invalid_argument("Sparse model requires nonempty positive subspaces");
    for (int i = 0; i < num_fields(); ++i) {
      fields_[i].validate();
      if (fields_[i].dimensions != dims) throw std::invalid_argument("Sparse model operators disagree on subspaces");
      if (i >= n_hyb_) validate_interaction_operator(fields_[i]);
      for (auto const &b : fields_[i].blocks)
        for (auto z : b.values)
          if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) throw std::invalid_argument("Nonfinite sparse model operator");
    }
    for (long p = 0; p < poles_; ++p)
      for (int i = 0; i < num_fields(); ++i)
        for (int j = 0; j < num_fields(); ++j)
          if (!std::isfinite(coeff(p, i, j).real()) || !std::isfinite(coeff(p, i, j).imag()))
            throw std::invalid_argument("Nonfinite sparse model coefficient");
    labels_.resize(fields_.size());
    within_set_.resize(fields_.size());
    sets_.reserve(fields_.size());
    for (auto [first, last] : {std::pair{0, n_hyb_}, std::pair{n_hyb_, num_fields()}}) {
      int start = sets_.size();
      for (int i = first; i < last; ++i) {
        int chosen = -1;
        for (int g = start; g < static_cast<int>(sets_.size()); ++g)
          if (fields_[sets_[g][0]].connection == fields_[i].connection) {
            chosen = g;
            break;
          }
        if (chosen == -1) {
          chosen = sets_.size();
          sets_.push_back({});
        }
        labels_[i]     = chosen;
        within_set_[i] = sets_[chosen].size();
        sets_[chosen].push_back(i);
      }
    }
    nda::vector<long> labels(fields_.size());
    for (int i = 0; i < num_fields(); ++i) labels(i) = labels_[i];
    check_sym_set_block_diagonal(coeff, labels, "Sparse operator model");
    storage_bound_ = sizeof(*this);
    storage_bound_ = checked_add(storage_bound_, vector_bytes(fields_));
    storage_bound_ = checked_add(storage_bound_, checked_multiply(checked_add(fields_.size(), checked_multiply(2, bars)), sizeof(SparseOperator)));
    storage_bound_ = checked_add(storage_bound_, coefficient_bytes);
    storage_bound_ = checked_add(storage_bound_, checked_add(vector_bytes(labels_), checked_add(vector_bytes(within_set_), vector_bytes(sets_))));
    for (auto const &set : sets_) storage_bound_ = checked_add(storage_bound_, vector_bytes(set));
    for (auto const &op : fields_) storage_bound_ = checked_add(storage_bound_, checked_add(operator_storage(op), transpose_storage(op)));
    // Bound every weighted family before constructing adjoints or barred arrays.
    for (long p = 0; p < poles_; ++p)
      for (int i = 0; i < num_fields(); ++i)
        for (bool dagger : {false, true}) {
          auto const &rep   = fields_[i];
          std::size_t bytes = metadata_storage(rep);
          for (std::size_t b = 0; b < rep.blocks.size(); ++b) {
            auto const &shape = rep.blocks[b];
            bytes             = checked_add(bytes, sizeof(long));
            if (rep.connection[b] == -1) continue;
            bytes = checked_add(bytes, checked_multiply(checked_extent(dagger ? shape.columns : shape.rows), sizeof(long)));
            for (int j : sets_[labels_[i]]) {
              complex weight = dagger ? coeff(p, j, i) : -coeff(p, i, j);
              if (weight != complex{})
                bytes = checked_add(bytes, checked_multiply(fields_[j].blocks[b].values.size(), sizeof(long) + sizeof(complex)));
            }
          }
          storage_bound_ = checked_add(storage_bound_, bytes);
        }
    if (storage_bound_ > budget) throw std::length_error("Sparse operator model exceeds owned storage budget");
    coefficients_ = nda::make_regular(coeff);
    adjoints_.reserve(fields_.size());
    for (auto const &op : fields_) adjoints_.push_back(op.adjoint());
    dagger_bars_.reserve(bars);
    reflected_bars_.reserve(bars);
    for (long p = 0; p < poles_; ++p)
      for (int i = 0; i < num_fields(); ++i) {
        std::vector<SparseOperator const *> direct, conjugate;
        std::vector<complex> forward, reflected;
        for (int j : sets_[labels_[i]]) {
          direct.push_back(&fields_[j]);
          conjugate.push_back(&adjoints_[j]);
          forward.push_back(coeff(p, j, i));
          reflected.push_back(-coeff(p, i, j));
        }
        dagger_bars_.push_back(combine(conjugate, forward));
        reflected_bars_.push_back(combine(direct, reflected));
      }
    if (host_storage_bytes() > storage_bound_) throw std::logic_error("Sparse model storage estimate underestimated ownership");
  }
  SparseOperator const &OperatorModel::field(int i, bool dagger) const {
    if (i < 0 || i >= num_fields()) throw std::out_of_range("Sparse model field index");
    return dagger ? adjoints_[i] : fields_[i];
  }
  SparseOperator const &OperatorModel::barred(int i, long pole, bool dagger) const {
    if (i < 0 || i >= num_fields() || pole < 0 || pole >= poles_) throw std::out_of_range("Sparse model barred index");
    auto index = checked_add(checked_multiply(checked_extent(pole), fields_.size()), i);
    return dagger ? dagger_bars_[index] : reflected_bars_[index];
  }
  std::size_t OperatorModel::host_storage_bytes() const {
    auto bytes = checked_add(sizeof(*this), checked_multiply(coefficients_.size(), sizeof(complex)));
    bytes      = checked_add(bytes, checked_add(vector_bytes(labels_), checked_add(vector_bytes(within_set_), vector_bytes(sets_))));
    for (auto const &set : sets_) bytes = checked_add(bytes, vector_bytes(set));
    for (auto const *family : {&fields_, &adjoints_, &dagger_bars_, &reflected_bars_}) {
      bytes = checked_add(bytes, vector_bytes(*family));
      for (auto const &op : *family) bytes = checked_add(bytes, operator_storage(op));
    }
    return bytes;
  }
  std::size_t OperatorModel::sparse_payload_bytes() const {
    std::size_t bytes = 0;
    for (auto const *family : {&fields_, &adjoints_, &dagger_bars_, &reflected_bars_})
      for (auto const &op : *family) bytes = checked_add(bytes, operator_payload(op));
    return bytes;
  }
#define EXPORT(C)                                                                                                                                    \
  OperatorModel export_model(triqs_xca::atom_diag::triqs_atom_diag_t<C> const &ad, nda::array_const_view<complex, 3> coeff, std::size_t budget) {    \
    return export_impl(ad, coeff, {}, nda::zeros<complex>(coeff.extent(0), 0, 0), budget);                                                           \
  }                                                                                                                                                  \
  OperatorModel export_model(triqs_xca::atom_diag::triqs_atom_diag_t<C> const &ad, nda::array_const_view<complex, 3> coeff,                          \
                             std::vector<triqs::operators::many_body_operator_complex> const &ops, nda::array_const_view<complex, 3> dyn,            \
                             std::size_t budget) {                                                                                                   \
    return export_impl(ad, coeff, ops, dyn, budget);                                                                                                 \
  }
  EXPORT(true)
  EXPORT(false)
#undef EXPORT
} // namespace triqs_xca::block_sparse::cuda

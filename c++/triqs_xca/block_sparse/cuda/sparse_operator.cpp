// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "sparse_operator.hpp"
#include <triqs/hilbert_space/imperative_operator.hpp>
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace triqs_xca::block_sparse::cuda {
  namespace {
    using complex = nda::dcomplex;
    using Space   = triqs::hilbert_space::hilbert_space;
    using Action  = triqs::hilbert_space::imperative_operator<Space, complex>;

    // TRIQS state<...,true>::foreach prunes through utility::is_zero. Export must
    // not do that: coefficients smaller than that threshold are still physics.
    struct ExactState {
      using value_type = complex;
      Space const *space;
      std::map<int, complex> amplitudes;
      explicit ExactState(Space const &hs) : space(&hs) {}
      Space const &get_hilbert() const { return *space; }
      complex &operator()(int index) { return amplitudes[index]; }
      template <typename F> friend void foreach (ExactState const &state, F f) {
        for (auto const &[index, value] : state.amplitudes)
          if (value != complex{}) f(index, value);
      }
    };

    void check_elements(long a, long b) {
      if (a < 0 || b < 0 || (a != 0 && b > std::numeric_limits<long>::max() / a))
        throw std::overflow_error("Sparse operator allocation extent overflow");
      auto elements = static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
      if (elements > std::numeric_limits<std::size_t>::max() / sizeof(complex)) throw std::overflow_error("Sparse operator allocation byte overflow");
    }

    template <bool C>
    SparseOperator export_impl(triqs_xca::atom_diag::triqs_atom_diag_t<C> const &ad, triqs::operators::many_body_operator_complex const &op) {
      // TRIQS's full Hilbert-space/state-action indices are ints, even though
      // individual Fock states are uint64_t. Check before its constructor shifts.
      if (ad.get_fops().size() >= std::numeric_limits<int>::digits)
        throw std::overflow_error("Sparse export exceeds TRIQS full Hilbert-space integer indexing");
      Space space(ad.get_fops());
      SparseOperator out;
      int nb = ad.n_subspaces();
      out.connection.assign(nb, -1);
      out.blocks.resize(nb);
      std::unordered_map<triqs::hilbert_space::fock_state_t, std::pair<int, long>> positions;
      for (int b = 0; b < nb; ++b) {
        auto const &states = ad.get_fock_states(b);
        out.dimensions.push_back(states.size());
        for (long j = 0; j < static_cast<long>(states.size()); ++j)
          if (!positions.emplace(states[j], std::pair{b, j}).second) throw std::invalid_argument("Sparse export: duplicate atom_diag Fock state");
      }
      struct Term {
        complex coefficient;
        Action action;
      };
      std::vector<Term> terms;
      for (auto const &term : op) {
        triqs::operators::many_body_operator_complex unit_monomial(complex{1.}, term.monomial);
        terms.push_back({term.coef, Action(unit_monomial, ad.get_fops())});
      }
      for (int b = 0; b < nb; ++b) {
        auto &block   = out.blocks[b];
        block.columns = out.dimensions[b];
        // Only sparse entries are held, never a rows*columns dense temporary.
        std::map<std::pair<long, long>, complex> entries;
        auto const &states = ad.get_fock_states(b);
        for (long column = 0; column < block.columns; ++column) {
          ExactState input(space);
          input(static_cast<int>(states[column])) = 1.;
          for (auto const &term : terms) {
            auto result = term.action(input);
            foreach (result, [&](int index, complex value) {
              auto found = positions.find(space.get_fock_state(index));
              // atom_diag applies the monomial in the full Fock space, then
              // projects its final state. Intermediate c/cdag connections are
              // insufficient for atom_diag instances with occupation cutoffs.
              if (found == positions.end()) return;
              int target = found->second.first;
              if (out.connection[b] != -1 && out.connection[b] != target)
                throw std::invalid_argument("Sparse export: monomials map source subspace " + std::to_string(b) + " to different targets");
              out.connection[b] = target;
              entries[{found->second.second, column}] += term.coefficient * value;
            });
          }
        }
        if (out.connection[b] == -1) continue;
        block.rows = out.dimensions[out.connection[b]];
        block.row_offsets.assign(block.rows + 1, 0);
        for (auto const &[position, value] : entries) {
          if (value == complex{}) continue; // Exact cancellation only.
          ++block.row_offsets[position.first + 1];
          block.column_indices.push_back(position.second);
          block.values.push_back(value);
        }
        for (long row = 0; row < block.rows; ++row) block.row_offsets[row + 1] += block.row_offsets[row];
        block.validate();
      }
      out.validate();
      return out;
    }
  } // namespace

  void SparseBlock::validate() const {
    if (rows < 0 || columns < 0 || row_offsets.empty() || static_cast<std::size_t>(rows) != row_offsets.size() - 1 || row_offsets.front() != 0
        || column_indices.size() != values.size() || values.size() > static_cast<std::size_t>(std::numeric_limits<long>::max())
        || row_offsets.back() != static_cast<long>(values.size()))
      throw std::invalid_argument("Invalid sparse operator CSR extents");
    for (long row = 0; row < rows; ++row) {
      long begin = row_offsets[row], end = row_offsets[row + 1], previous = -1;
      if (begin < 0 || end < begin || end > static_cast<long>(values.size())) throw std::invalid_argument("Invalid sparse operator row offsets");
      for (long j = begin; j < end; ++j) {
        if (column_indices[j] <= previous || column_indices[j] >= columns) throw std::invalid_argument("Invalid sparse operator column order/index");
        previous = column_indices[j];
      }
    }
  }
  nda::matrix<complex> SparseBlock::to_dense() const {
    validate();
    check_elements(rows, columns);
    nda::matrix<complex> out = nda::zeros<complex>(rows, columns);
    for (long row = 0; row < rows; ++row)
      for (long j = row_offsets[row]; j < row_offsets[row + 1]; ++j) out(row, column_indices[j]) = values[j];
    return out;
  }
  SparseBlock SparseBlock::adjoint() const {
    validate();
    if (columns == std::numeric_limits<long>::max()) throw std::overflow_error("Sparse transpose row count overflow");
    SparseBlock out;
    out.rows    = columns;
    out.columns = rows;
    out.row_offsets.assign(columns + 1, 0);
    for (auto col : column_indices) ++out.row_offsets[col + 1];
    for (long row = 0; row < columns; ++row) out.row_offsets[row + 1] += out.row_offsets[row];
    auto cursor = out.row_offsets;
    out.column_indices.resize(values.size());
    out.values.resize(values.size());
    for (long row = 0; row < rows; ++row)
      for (long j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
        long dst                = cursor[column_indices[j]]++;
        out.column_indices[dst] = row;
        out.values[dst]         = std::conj(values[j]);
      }
    return out;
  }
  nda::array<complex, 3> SparseBlock::apply_left(nda::array_const_view<complex, 3> input) const {
    validate();
    if (input.extent(1) != columns) throw std::invalid_argument("Sparse left apply shape mismatch");
    check_elements(rows, input.extent(2));
    check_elements(input.extent(0), rows * input.extent(2));
    nda::array<complex, 3> out = nda::zeros<complex>(input.extent(0), rows, input.extent(2));
    for (long t = 0; t < input.extent(0); ++t)
      for (long row = 0; row < rows; ++row)
        for (long j = row_offsets[row]; j < row_offsets[row + 1]; ++j)
          for (long col = 0; col < input.extent(2); ++col) out(t, row, col) += values[j] * input(t, column_indices[j], col);
    return out;
  }
  nda::array<complex, 3> SparseBlock::apply_right(nda::array_const_view<complex, 3> input) const {
    validate();
    if (input.extent(2) != rows) throw std::invalid_argument("Sparse right apply shape mismatch");
    check_elements(input.extent(1), columns);
    check_elements(input.extent(0), input.extent(1) * columns);
    nda::array<complex, 3> out = nda::zeros<complex>(input.extent(0), input.extent(1), columns);
    for (long t = 0; t < input.extent(0); ++t)
      for (long row = 0; row < rows; ++row)
        for (long j = row_offsets[row]; j < row_offsets[row + 1]; ++j)
          for (long i = 0; i < input.extent(1); ++i) out(t, i, column_indices[j]) += input(t, i, row) * values[j];
    return out;
  }
  void SparseOperator::validate() const {
    if (dimensions.size() != connection.size() || blocks.size() != connection.size())
      throw std::invalid_argument("Invalid sparse operator metadata extents");
    for (std::size_t b = 0; b < connection.size(); ++b) {
      blocks[b].validate();
      int target = connection[b];
      if (dimensions[b] < 0 || target < -1 || (target >= 0 && static_cast<std::size_t>(target) >= dimensions.size()))
        throw std::invalid_argument("Invalid sparse operator dimensions/connection");
      if (blocks[b].columns != dimensions[b] || blocks[b].rows != (target == -1 ? 0 : dimensions[target]))
        throw std::invalid_argument("Sparse operator block dimensions disagree with metadata");
    }
  }
  SparseOperator SparseOperator::adjoint() const {
    validate();
    SparseOperator out{dimensions, std::vector<int>(connection.size(), -1), std::vector<SparseBlock>(connection.size())};
    for (std::size_t b = 0; b < connection.size(); ++b) out.blocks[b].columns = dimensions[b];
    for (std::size_t b = 0; b < connection.size(); ++b) {
      int target = connection[b];
      if (target == -1) continue;
      if (target < 0 || static_cast<std::size_t>(target) >= connection.size() || out.connection[target] != -1)
        throw std::invalid_argument("Sparse operator adjoint requires an injective connection map");
      out.connection[target] = b;
      out.blocks[target]     = blocks[b].adjoint();
    }
    return out;
  }
  std::size_t SparseOperator::nonzeros() const {
    std::size_t count = 0;
    for (auto const &block : blocks) {
      if (block.values.size() > std::numeric_limits<std::size_t>::max() - count) throw std::overflow_error("Sparse operator nnz overflow");
      count += block.values.size();
    }
    return count;
  }
  void validate_interaction_operator(SparseOperator const &op) {
    op.validate();
    if (std::none_of(op.connection.begin(), op.connection.end(), [](int b) { return b != -1; }))
      throw std::invalid_argument("Sparse interaction operator is zero on every atom_diag subspace");
    std::vector<bool> seen(op.connection.size());
    for (int target : op.connection) {
      if (target == -1) continue;
      if (seen[target]) throw std::invalid_argument("Sparse interaction operator requires an injective connection map");
      seen[target] = true;
    }
  }
  SparseOperator export_operator(triqs_xca::atom_diag::triqs_atom_diag_t<true> const &ad, triqs::operators::many_body_operator_complex const &op) {
    return export_impl(ad, op);
  }
  SparseOperator export_operator(triqs_xca::atom_diag::triqs_atom_diag_t<false> const &ad, triqs::operators::many_body_operator_complex const &op) {
    return export_impl(ad, op);
  }
} // namespace triqs_xca::block_sparse::cuda

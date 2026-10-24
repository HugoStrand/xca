#pragma once
#include "triqs_xca/action.hpp"
#include "atom_diag.hpp"
#include "dynint.hpp"

namespace triqs_xca::block_sparse {
  nda::vector<long> operator_block_dims(BlockOpSymQuartet const &operators);
  void check_operator_block_dims(BlockOpSymQuartet const &operators, nda::vector_const_view<long> dims);

  inline Action<BlockOpSymQuartet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
                                              nda::array_const_view<nda::dcomplex, 3> coefficients, BlockOpSymQuartet const &operators,
                                              std::vector<triqs_xca::Statistics> statistics, int n_hyb, nda::vector<long> dims) {
    if (nda::sum(operators.sym_set_sizes) != statistics.size()) throw std::invalid_argument("Action: statistics must cover every operator");
    check_operator_block_dims(operators, dims);
    return {std::move(mesh), poles, coefficients, std::move(statistics), n_hyb, std::move(dims),
            [&](auto const &c) { return BlockOpSymQuartet(operators.Fs, operators.F_dags, c, operators.sym_set_labels); }};
  }

  inline Action<BlockOpSymQuartet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
                                              nda::array_const_view<nda::dcomplex, 3> coefficients, BlockOpSymQuartet const &operators,
                                              int n_int = 0) {
    int n = nda::sum(operators.sym_set_sizes);
    return make_action(std::move(mesh), poles, coefficients, operators, action_statistics(n, n - n_int), n - n_int,
                       operator_block_dims(operators));
  }

  template <bool IsComplex> nda::vector<long> atom_block_dims(triqs::atom_diag::atom_diag<IsComplex> const &ad) {
    auto const &dims = ad.get_subspace_dims();
    nda::vector<long> result(dims.size());
    for (long b = 0; b < result.size(); ++b) result(b) = dims[b];
    return result;
  }

  template <bool IsComplex>
  Action<BlockOpSymQuartet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
                                       nda::array_const_view<nda::dcomplex, 3> coefficients, triqs::atom_diag::atom_diag<IsComplex> const &ad) {
    int n = ad.get_fops().size();
    return {std::move(mesh), poles, coefficients, action_statistics(n, n), n, atom_block_dims(ad),
            [&](auto const &c) { return std::get<0>(atom_diag::get_operators(ad, c)); }};
  }

  template <bool IsComplex>
  Action<BlockOpSymQuartet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
      nda::array_const_view<nda::dcomplex, 3> hyb_coeffs, triqs::atom_diag::atom_diag<IsComplex> const &ad,
      std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
      nda::array_const_view<nda::dcomplex, 3> dynint_coeffs) {
    int h = ad.get_fops().size(), n = h + dynint_ops.size();
    if (hyb_coeffs.extent(1) != h || dynint_coeffs.extent(1) != dynint_ops.size())
      throw std::invalid_argument("Action: coefficient blocks must match fundamental operators and dynint_ops");
    auto coefficients = get_extended_coefficients(hyb_coeffs, dynint_coeffs);
    return {std::move(mesh), poles, coefficients, action_statistics(n, h), h, atom_block_dims(ad),
            [&](auto const &c) { return std::get<0>(dynint::get_operators_and_interactions(ad, c, dynint_ops)); }};
  }
}

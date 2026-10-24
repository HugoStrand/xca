#pragma once
#include "triqs_xca/action.hpp"
#include "atom_diag.hpp"
#include "dynint.hpp"

namespace triqs_xca::dense {
  /// Rebuild bars; supplied precomputed contractions need not match the coefficients.
  inline Action<FSet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
                                  nda::array_const_view<nda::dcomplex, 3> coefficients, FSet const &operators,
                                  std::vector<diagram::Statistics> statistics, int n_hyb) {
    auto const &F = operators.Fs;
    if (F.extent(0) != statistics.size() || F.extent(1) != F.extent(2) || operators.F_dags.shape() != F.shape())
      throw std::invalid_argument("Action: inconsistent dense operator shapes");
    return {std::move(mesh), poles, coefficients, std::move(statistics), n_hyb, nda::vector<long>{F.extent(1)},
            [&](auto const &c) { return FSet(F, operators.F_dags, c); }};
  }

  inline Action<FSet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
                                  nda::array_const_view<nda::dcomplex, 3> coefficients, FSet const &operators) {
    int n = operators.Fs.extent(0);
    return make_action(std::move(mesh), poles, coefficients, operators, action_statistics(n, n), n);
  }

  template <bool IsComplex>
  Action<FSet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
                           nda::array_const_view<nda::dcomplex, 3> coefficients, triqs::atom_diag::atom_diag<IsComplex> const &ad) {
    int n = ad.get_fops().size();
    return {std::move(mesh), poles, coefficients, action_statistics(n, n), n, nda::vector<long>{ad.get_full_hilbert_space_dim()},
            [&](auto const &c) { return atom_diag::get_operators(ad, c); }};
  }

  template <bool IsComplex>
  Action<FSet> make_action(triqs::mesh::dlr_imtime mesh, nda::vector_const_view<double> poles,
      nda::array_const_view<nda::dcomplex, 3> hyb_coeffs, triqs::atom_diag::atom_diag<IsComplex> const &ad,
      std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops,
      nda::array_const_view<nda::dcomplex, 3> dynint_coeffs) {
    int h = ad.get_fops().size(), n = h + dynint_ops.size();
    if (hyb_coeffs.extent(1) != h || dynint_coeffs.extent(1) != dynint_ops.size())
      throw std::invalid_argument("Action: coefficient blocks must match fundamental operators and dynint_ops");
    auto coefficients = hyb::get_extended_coefficients(hyb_coeffs, dynint_coeffs);
    return {std::move(mesh), poles, coefficients, action_statistics(n, h), h, nda::vector<long>{ad.get_full_hilbert_space_dim()},
            [&](auto const &c) { return dynint::get_operators_and_interactions(ad, c, dynint_ops); }};
  }
}

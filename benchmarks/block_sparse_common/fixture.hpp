#pragma once
#include <triqs_xca/block_sparse/cuda/path.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <triqs/atom_diag.hpp>

namespace block_sparse_bench {
  using nda::dcomplex;
  using triqs::operators::many_body_operator_complex;
  namespace gpu = triqs_xca::block_sparse::cuda;
  inline triqs::atom_diag::atom_diag<true> atom(int n, std::string const &partition) {
    using triqs::operators::c;
    using triqs::operators::c_dag;
    triqs::atom_diag::fundamental_operator_set fops;
    many_body_operator_complex H, Q, A, B;
    for (int i = 0; i < n; ++i) {
      fops.insert("f", i);
      auto number = triqs::operators::n("f", i);
      Q += number;
      (i < n / 2 ? A : B) += number;
      H += (.13 * (i + 1) + .007 * i * i) * number;
      for (int j = 0; j < i; ++j) {
        H += (.17 + .003 * (i + j)) * number * triqs::operators::n("f", j);
        if (partition == "occupation" || (partition == "mixed" && (i < n / 2) != (j < n / 2))) continue;
        dcomplex z{.011 * (1 + i + j), .009 * (i - j)};
        H += z * c_dag("f", i) * c("f", j) + std::conj(z) * c_dag("f", j) * c("f", i);
      }
    }
    if (partition == "occupation") return {H, fops};
    return {H, fops, partition == "number" ? std::vector<many_body_operator_complex>{Q} : std::vector<many_body_operator_complex>{A, B}};
  }
  inline nda::array<dcomplex, 3> coefficients(triqs::atom_diag::atom_diag<true> const &ad) {
    int n                         = ad.get_fops().size();
    nda::array<dcomplex, 3> coeff = nda::zeros<dcomplex>(2, n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        bool compatible = true;
        for (int b = 0; b < ad.n_subspaces(); ++b) compatible = compatible && ad.c_connection(i, b) == ad.c_connection(j, b);
        if (compatible)
          for (int p = 0; p < 2; ++p) coeff(p, i, j) = i == j ? dcomplex{.3 - .1 * p} : dcomplex{.025 * std::cos(i - j), .025 * std::sin(i - j)};
      }
    return coeff;
  }
  inline std::vector<long> analytic_dimensions(int n, std::string const &partition) {
    auto binomial = [](int n, int k) {
      std::size_t value = 1;
      for (int i = 1; i <= k; ++i) value = gpu::checked_multiply(value, n - i + 1) / i;
      if (value > std::size_t(std::numeric_limits<long>::max())) throw std::overflow_error("Sector dimension overflow");
      return static_cast<long>(value);
    };
    std::vector<long> dims;
    if (partition == "occupation")
      dims.assign(std::size_t{1} << n, 1);
    else if (partition == "number")
      for (int q = 0; q <= n; ++q) dims.push_back(binomial(n, q));
    else
      for (int a = 0; a <= n / 2; ++a)
        for (int b = 0; b <= n - n / 2; ++b) dims.push_back(gpu::checked_multiply(binomial(n / 2, a), binomial(n - n / 2, b)));
    return dims;
  }
} // namespace block_sparse_bench

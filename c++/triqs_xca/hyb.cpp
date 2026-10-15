#include "triqs_xca/hyb.hpp"

using nda::linalg::matmul;

using cppdlr::_;

namespace triqs_xca::hyb {

  nda::array<dcomplex, 3> coefs2vals(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles) {
    auto dlr_rf = cppdlr::build_dlr_rf(Lambda, eps);
    auto itops  = cppdlr::imtime_ops(Lambda, dlr_rf);
    return coefs2vals(beta, itops, coefs, poles);
  }

  nda::array<dcomplex, 3> coefs2vals(double beta, const cppdlr::imtime_ops &itops, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles) {
    long n1     = coefs.extent(1);
    long n2     = coefs.extent(2);
    int r       = itops.rank();
    auto dlr_it = itops.get_itnodes();
    int p       = static_cast<int>(poles.size());
    auto kmat   = cppdlr::build_k_it(dlr_it, nda::make_regular(beta * poles));
    auto cf_r   = nda::reshape(nda::make_regular(coefs), p, coefs.size() / p); // reshape needs C-ordered memory, coefs may have any strides
    nda::array<dcomplex, 3> vals(r, n1, n2);
    reshape(vals, r, n1 * n2) = matmul(kmat, cf_r);
    return vals;
  }

  nda::array<dcomplex, 3> get_extended_coefficients(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> dynint_coeffs) {

    assert(hyb_coeffs.extent(0) == dynint_coeffs.extent(0));
    assert(hyb_coeffs.extent(1) == hyb_coeffs.extent(2));
    assert(dynint_coeffs.extent(1) == dynint_coeffs.extent(2));

    int r        = hyb_coeffs.extent(0);
    int n_hyb    = hyb_coeffs.extent(1);
    int n_dynint = dynint_coeffs.extent(1);
    int n_ext    = n_hyb + n_dynint;

    auto ext_coeffs = nda::array<dcomplex, 3>::zeros(r, n_ext, n_ext);

    ext_coeffs(_, nda::range(0, n_hyb), nda::range(0, n_hyb))         = hyb_coeffs;
    ext_coeffs(_, nda::range(n_hyb, n_ext), nda::range(n_hyb, n_ext)) = dynint_coeffs;

    return ext_coeffs;
  }

} // namespace triqs_xca::hyb

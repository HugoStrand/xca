#include "triqs_xca/hyb.hpp"

using nda::linalg::matmul;

using cppdlr::_;

namespace triqs_xca::hyb {

  /**
     * @brief Constructor for FSet
     * @param[in] tau_mesh TRIQS imaginary time DLR mesh
     * @param[in] hyb_poles poles of the hybridization
     * @param[in] hyb_coeffs pole coefficients of the hybridization
     */
  Hybridization::Hybridization(const triqs::mesh::dlr_imtime &tau_mesh, nda::vector_const_view<double> hyb_poles,
                               nda::array_const_view<dcomplex, 3> hyb_coeffs, double refl_sign)
     : tau_mesh(tau_mesh),
       poles(hyb_poles * tau_mesh.beta()),
       coeffs(hyb_coeffs),
       values(coefs2vals(tau_mesh.beta(), tau_mesh.dlr_it(), hyb_coeffs, hyb_poles)),
       // Follow sign convention of block_sparse/diagram_evaluator for reflected hybridization function.
       values_reflect(refl_sign * tau_mesh.dlr_it().reflect(values)),
       k_0_p(poles.size()),
       k_0_m(poles.size()),
       k_it_p(tau_mesh.dlr_it().rank(), poles.size()),
       k_it_m(tau_mesh.dlr_it().rank(), poles.size())
  {

    // Precompute kernels at tau=0 and at imaginary time nodes for both +poles and -poles,
    // since these are needed frequently in the diagram evaluation.

    auto dlr_it = tau_mesh.dlr_it().get_itnodes();

    for (size_t p = 0; p < poles.size(); ++p) {

      k_0_p(p) = cppdlr::k_it(0, +poles(p));
      k_0_m(p) = cppdlr::k_it(0, -poles(p));

      for (int t = 0; t < tau_mesh.dlr_it().rank(); ++t) {
        k_it_p(t, p) = cppdlr::k_it(dlr_it(t), +poles(p));
        k_it_m(t, p) = cppdlr::k_it(dlr_it(t), -poles(p));
      }
    }
  };


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

#pragma once

#include <nda/nda.hpp>

#include <triqs/gfs.hpp>


namespace triqs_xca::hyb {

  using nda::dcomplex;

  /**
 * @brief Convert hybridization coefficients to values at DLR imaginary time nodes
 * @param[in] beta inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] coefs DLR coefficients
 * @param[in] poles DLR poles
 */
  nda::array<dcomplex, 3> coefs2vals(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles);

  /**
 * @brief Convert hybridization coefficients to values at DLR imaginary time nodes
 * @param[in] itos cppdlr imaginary time object
 * @param[in] coefs DLR coefficients
 * @param[in] poles DLR poles
 */
  nda::array<dcomplex, 3> coefs2vals(double beta, const cppdlr::imtime_ops &itops, nda::array_const_view<dcomplex, 3> coefs,
                                     nda::vector_const_view<double> poles);

  /**
   * @brief Extend the hybridization coefficients block-diagonally by the dynamical-interaction coefficients
   * @param[in] hyb_coeffs hybridization coefficients, shape (p, n_hyb, n_hyb)
   * @param[in] dynint_coeffs dynamical-interaction coefficients, shape (p, n_int, n_int)
   * @return extended coefficients, shape (p, n_hyb + n_int, n_hyb + n_int), zero in the cross blocks
   */
  nda::array<dcomplex, 3> get_extended_coefficients(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> dynint_coeffs);

} // namespace triqs_xca::hyb
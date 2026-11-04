#pragma once

#include <nda/nda.hpp>

namespace triqs_xca::dense {

  /// Fixed-size, non-conjugated complex matrix product. Accepts strided NDA views;
  /// all views must have shape (N, N), and C must not overlap A or B.
  /// beta=0 does not read C. Intended for finite inputs.
  template <int N>
  inline void small_gemm(nda::dcomplex alpha, nda::array_const_view<nda::dcomplex, 2> A, nda::array_const_view<nda::dcomplex, 2> B,
                         nda::dcomplex beta, nda::array_view<nda::dcomplex, 2> C) {
    static_assert(N > 0);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        double real = 0., imag = 0.;
        for (int k = 0; k < N; ++k) {
          auto a = A(i, k), b = B(k, j);
          real += a.real() * b.real() - a.imag() * b.imag();
          imag += a.real() * b.imag() + a.imag() * b.real();
        }
        nda::dcomplex value{real, imag};
        C(i, j) = beta == nda::dcomplex{} ? alpha * value : alpha * value + beta * C(i, j);
      }
  }

} // namespace triqs_xca::dense

#pragma once

#include <nda/nda.hpp>
#include <Eigen/Core>

namespace triqs_xca::dense {

  /// Handwritten reference for the fixed-size product. Accepts strided row-major NDA views;
  /// all views must have shape (N, N), and C must not overlap A or B.
  /// beta=0 does not read C. Intended for finite inputs.
  template <int N>
  inline void small_gemm_handwritten(nda::dcomplex alpha, nda::array_const_view<nda::dcomplex, 2> A, nda::array_const_view<nda::dcomplex, 2> B,
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

  namespace small_gemm_detail {
    template <typename Left, typename Right, typename Output>
    inline void eigen_product(nda::dcomplex alpha, Left const &A, Right const &B, nda::dcomplex beta, Output &C) {
      if (beta == nda::dcomplex{}) {
        C.noalias() = alpha * A * B;
      } else {
        if (beta != nda::dcomplex{1.}) C *= beta;
        C.noalias() += alpha * A * B;
      }
    }
  } // namespace small_gemm_detail

  /// Fixed-size, non-conjugated Eigen matrix product. Accepts strided row-major NDA views;
  /// all views must have shape (N, N), and C must not overlap A or B.
  /// beta=0 does not read C. Intended for finite inputs.
  template <int N>
  inline void small_gemm(nda::dcomplex alpha, nda::array_const_view<nda::dcomplex, 2> A, nda::array_const_view<nda::dcomplex, 2> B,
                         nda::dcomplex beta, nda::array_view<nda::dcomplex, 2> C) {
    static_assert(N > 0);
    using Matrix         = Eigen::Matrix<nda::dcomplex, N, N, Eigen::RowMajor>;
    auto const &a_stride = A.indexmap().strides(), &b_stride = B.indexmap().strides(), &c_stride = C.indexmap().strides();
    auto packed = [](auto const &strides) { return strides[0] == N && strides[1] == 1; };

    // Evaluator operands are packed. Keep their unit column stride visible to
    // Eigen at compile time; general NDA slices use the explicit-stride path.
    if (packed(a_stride) && packed(b_stride) && packed(c_stride)) {
      Eigen::Map<Matrix const, Eigen::Unaligned> a(A.data()), b(B.data());
      Eigen::Map<Matrix, Eigen::Unaligned> c(C.data());
      small_gemm_detail::eigen_product(alpha, a, b, beta, c);
    } else {
      using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
      Eigen::Map<Matrix const, Eigen::Unaligned, Stride> a(A.data(), Stride{a_stride[0], a_stride[1]});
      Eigen::Map<Matrix const, Eigen::Unaligned, Stride> b(B.data(), Stride{b_stride[0], b_stride[1]});
      Eigen::Map<Matrix, Eigen::Unaligned, Stride> c(C.data(), Stride{c_stride[0], c_stride[1]});
      // Eigen 5's strided complex GEMM can use aligned packet stores even for
      // Unaligned maps. A stack result also accepts destinations aligned to
      // only alignof(dcomplex), while preserving beta=0's overwrite semantics.
      Matrix result;
      if (beta != nda::dcomplex{}) result = c;
      small_gemm_detail::eigen_product(alpha, a, b, beta, result);
      c = result;
    }
  }

} // namespace triqs_xca::dense

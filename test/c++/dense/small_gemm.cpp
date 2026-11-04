#include "triqs_xca/dense/small_gemm.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
  using nda::dcomplex;

  template <int N> void check_products() {
    // Exercise the strided-view contract as well as packed evaluator matrices.
    for (int stride : {1, 2}) {
      nda::matrix<dcomplex> a_store(2 * N, 2 * N), b_store(2 * N, 2 * N), c_store(2 * N, 2 * N);
      a_store() = b_store() = c_store() = dcomplex{37., -19.};
      auto rows                         = nda::range(0, stride * N, stride);
      auto A = a_store(rows, rows), B = b_store(rows, rows), C = c_store(rows, rows);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          A(i, j) = {0.12 * (i + 1) - 0.04 * j, -0.23 + 0.19 * i + 0.07 * j};
          B(i, j) = {-0.18 * i + 0.09 * j, 0.21 - 0.11 * i + 0.05 * j};
        }
      auto product = nda::linalg::matmul(nda::make_regular(A), nda::make_regular(B));
      for (auto [alpha, beta] : {std::pair{dcomplex{1.}, dcomplex{}}, std::pair{dcomplex{1.}, dcomplex{1.}},
                                 std::pair{dcomplex{0.2, -0.3}, dcomplex{-0.4, 0.1}}, std::pair{dcomplex{}, dcomplex{1.}}}) {
        C()                            = dcomplex{-0.17, 0.29};
        nda::matrix<dcomplex> expected = alpha * product + beta * C;
        if (beta == dcomplex{}) {
          // Overwrite must not read stale or uninitialized destination values.
          double nan = std::numeric_limits<double>::quiet_NaN();
          C()        = dcomplex{nan, nan};
        }
        triqs_xca::dense::small_gemm<N>(alpha, A, B, beta, C);
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j) EXPECT_LE(std::abs(C(i, j) - expected(i, j)), 2e-13 * std::max(1., std::abs(expected(i, j))));
        EXPECT_EQ(c_store(2 * N - 1, 2 * N - 1), (dcomplex{37., -19.}));
      }
    }
  }
} // namespace

TEST(DenseSmallGemm, complex_overwrite_accumulation_and_strides) {
  check_products<2>();
  check_products<4>();
  check_products<8>();
}

#include "triqs_xca/dense/small_gemm.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <limits>

namespace {
  using nda::dcomplex;

  template <int N, typename AView, typename BView, typename CView> void check_products(AView A, BView B, CView C) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        A(i, j) = {0.12 * (i + 1) - 0.04 * j, -0.23 + 0.19 * i + 0.07 * j};
        B(i, j) = {-0.18 * i + 0.09 * j, 0.21 - 0.11 * i + 0.05 * j};
      }
    auto product = nda::linalg::matmul(nda::make_regular(A), nda::make_regular(B));
    for (bool handwritten : {false, true}) {
      SCOPED_TRACE(handwritten ? "handwritten" : "Eigen");
      for (auto [alpha, beta] : {std::pair{dcomplex{1.}, dcomplex{}}, std::pair{dcomplex{1.}, dcomplex{1.}},
                                 std::pair{dcomplex{0.2, -0.3}, dcomplex{-0.4, 0.1}}, std::pair{dcomplex{}, dcomplex{1.}},
                                 std::pair{dcomplex{}, dcomplex{}}}) {
        C()                            = dcomplex{-0.17, 0.29};
        nda::matrix<dcomplex> expected = alpha * product + beta * C;
        if (beta == dcomplex{}) {
          // Overwrite must not read stale or uninitialized destination values.
          double nan = std::numeric_limits<double>::quiet_NaN();
          C()        = dcomplex{nan, nan};
        }
        if (handwritten)
          triqs_xca::dense::small_gemm_handwritten<N>(alpha, A, B, beta, C);
        else
          triqs_xca::dense::small_gemm<N>(alpha, A, B, beta, C);
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j) EXPECT_LE(std::abs(C(i, j) - expected(i, j)), 2e-13 * std::max(1., std::abs(expected(i, j))));
      }
    }
  }

  template <int N> void check_layouts() {
    SCOPED_TRACE(N);
    nda::matrix<dcomplex> A(N, N), B(N, N), C(N, N);
    check_products<N>(A(), B(), C());

    // The NDA API accepts row-major views, including noncontiguous slices.
    // Different row/column strides check the Eigen::Map adapter beyond packing.
    nda::matrix<dcomplex> a_store(4 * N, 4 * N), b_store(4 * N, 4 * N), c_store(4 * N, 4 * N);
    a_store() = b_store() = c_store() = dcomplex{37., -19.};
    auto packed = nda::range(N), rows = nda::range(0, 2 * N, 2), cols = nda::range(0, 3 * N, 3);
    check_products<N>(a_store(packed, packed), b_store(packed, packed), c_store(packed, packed));
    check_products<N>(a_store(rows, cols), b_store(cols, rows), c_store(rows, cols));
    EXPECT_EQ(c_store(4 * N - 1, 4 * N - 1), (dcomplex{37., -19.}));
  }

  template <int N> void check_natural_alignment() {
    SCOPED_TRACE(N);
    // Complex<double> storage can be aligned to eight bytes without satisfying
    // Eigen's sixteen-byte SIMD alignment. Heap allocations often hide this.
    struct alignas(16) Storage {
      double offset;
      std::array<dcomplex, 4 * N * N> data;
    };
    Storage as{}, bs{}, cs{};
    for (auto *storage : {&as, &bs, &cs}) {
      ASSERT_EQ(reinterpret_cast<std::uintptr_t>(storage->data.data()) % 16, std::uintptr_t{8});
      storage->data.fill(dcomplex{37., -19.});
    }
    nda::array_view<dcomplex, 2> A{{N, N}, as.data.data()}, B{{N, N}, bs.data.data()}, C{{N, N}, cs.data.data()};
    check_products<N>(A, B, C);
    nda::array_view<dcomplex, 2> As{{2 * N, 2 * N}, as.data.data()}, Bs{{2 * N, 2 * N}, bs.data.data()}, Cs{{2 * N, 2 * N}, cs.data.data()};
    auto every_other = nda::range(0, 2 * N, 2);
    check_products<N>(As(every_other, every_other), Bs(every_other, every_other), Cs(every_other, every_other));
    EXPECT_EQ(cs.data.back(), (dcomplex{37., -19.}));
  }
} // namespace

TEST(DenseSmallGemm, complex_overwrite_accumulation_and_strides) {
  check_layouts<2>();
  check_layouts<4>();
  check_layouts<8>();
  check_layouts<16>();
}

TEST(DenseSmallGemm, natural_complex_alignment) {
  check_natural_alignment<2>();
  check_natural_alignment<4>();
  check_natural_alignment<8>();
  check_natural_alignment<16>();
}

#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
#ifdef NDA_HAVE_CUDA
#include <cppdlr/cppdlr.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include <nda/lapack.hpp>
#include <limits>

namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using nda::dcomplex;
  constexpr auto all = nda::range::all;
  nda::array<dcomplex, 3> coefficients(long r, long m, long n, double phase) {
    nda::array<dcomplex, 3> out(r, m, n);
    for (long t = 0; t < r; ++t)
      for (long i = 0; i < m; ++i)
        for (long j = 0; j < n; ++j)
          out(t, i, j) = dcomplex{std::sin(.7 * t + .3 * i - .2 * j + phase), std::cos(.1 * t - .4 * i + .8 * j + phase)} / (r * std::max(m, n));
    return out;
  }
  void compare(gpu::TensorConstView actual, nda::array_const_view<dcomplex, 3> expected, double tolerance = 1e-12) {
    auto a = nda::to_host(actual);
    ASSERT_EQ(a.shape(), expected.shape());
    EXPECT_LE(nda::max_element(nda::abs(a - expected)), tolerance);
  }
  cppdlr::imtime_ops small_system() {
    nda::vector<double> rf{-2., 0., 2.}, nodes{0., .4, -.1};
    nda::matrix<double> C(3, 3);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) C(i, j) = cppdlr::k_it(nodes(i), rf(j));
    auto lu = C;
    nda::vector<int> pivots(3);
    if (nda::lapack::getrf(lu, pivots) != 0) throw std::runtime_error("Test LU failure");
    return cppdlr::imtime_ops(4., rf, nodes, C, lu, pivots);
  }
} // namespace
TEST(BlockSparseCudaDlr, RectangularCachedSolvesAndVariableRhs) {
  auto host = small_system();
  auto ops  = std::make_shared<gpu::DeviceOps>();
  gpu::DlrOps device(ops, host);
  auto tables = device.storage_bytes();
  EXPECT_EQ(device.rank(), 3);
  for (auto [m, n] : std::vector<std::pair<long, long>>{{1, 1}, {3, 5}, {2, 7}, {3, 5}}) {
    SCOPED_TRACE(::testing::Message() << m << "x" << n);
    auto cf = coefficients(3, m, n, .3), values = host.coefs2vals(cf);
    gpu::Tensor in(values), out(3, m, n), reconstructed(3, m, n);
    device.vals2coefs(in, out);
    compare(out, cf);
    compare(out, host.vals2coefs(values));
    device.coefs2vals(out, reconstructed);
    compare(reconstructed, values);
    auto resident = device.storage_bytes();
    EXPECT_EQ(resident, tables + 3 * m * n * sizeof(dcomplex));
    device.vals2coefs(in, in);
    compare(in, cf);
    EXPECT_EQ(device.storage_bytes(), resident);
    device.reflect(reconstructed, out);
    compare(out, host.reflect(values));
  }
}
TEST(BlockSparseCudaDlr, RealMeshesRectangularConvolutionAndComposedEdge) {
  for (auto [lambda, eps] : std::vector<std::pair<double, double>>{{10., 1e-10}, {40., 1e-12}, {120., 1e-12}})
    for (bool sym : {false, true}) {
      SCOPED_TRACE(::testing::Message() << lambda << " " << eps << " " << sym);
      cppdlr::imtime_ops host(lambda, cppdlr::build_dlr_rf(lambda, eps, sym), sym);
      auto ops = std::make_shared<gpu::DeviceOps>();
      gpu::DlrOps device(ops, host);
      long r = host.rank();
      for (auto [M, K, N] : std::vector<std::tuple<long, long, long>>{{1, 1, 1}, {3, 5, 2}, {5, 5, 3}, {3, 5, 2}}) {
        auto fc = coefficients(r, M, K, .1), gc = coefficients(r, K, N, .9);
        gpu::Tensor fd(fc), gd(gc), out(r, M, N);
        for (double beta : {.5, 2., 7.}) {
          device.convolve(beta, fd, gd, out);
          compare(out, host.convolve(beta, fc, gc, cppdlr::TIME_ORDERED));
        }
        auto resident = device.storage_bytes();
        device.convolve(2., fd, gd, out);
        EXPECT_EQ(device.storage_bytes(), resident);
        auto fv = host.coefs2vals(fc), gv = host.coefs2vals(gc);
        gpu::Tensor fvd(fv), gvd(gv), fs(r, M, K), gs(r, K, N);
        device.vals2coefs(fvd, fs);
        device.vals2coefs(gvd, gs);
        gpu::Tensor back(r, M, K);
        device.coefs2vals(fs, back);
        compare(back, fv);
        device.convolve(2., fs, gs, out);
        compare(out, host.convolve(2., host.vals2coefs(fv), host.vals2coefs(gv), cppdlr::TIME_ORDERED));
      }
    }
}
TEST(BlockSparseCudaDlr, AliasesCancellationAndBudgetRecovery) {
  auto host = small_system();
  auto ops  = std::make_shared<gpu::DeviceOps>();
  gpu::DlrOps device(ops, host);
  auto fc = coefficients(3, 3, 3, .3), gc = coefficients(3, 3, 2, .8);
  fc(1, all, all) = -fc(0, all, all) + dcomplex{1e-11, -2e-11} * fc(2, all, all);
  gpu::Tensor fd(fc), gd(gc), out(3, 3, 2);
  auto expected = host.convolve(3., fc, gc, cppdlr::TIME_ORDERED);
  device.convolve(3., fd, gd, gd);
  compare(gd, expected);
  gd = gc;
  device.convolve(3., fd, gd, out);
  compare(out, expected);
  auto tables = gpu::DlrOps(ops, host).storage_bytes();
  EXPECT_THROW(gpu::DlrOps(ops, host, tables - 1), std::length_error);
  gpu::DlrOps limited(ops, host, tables);
  EXPECT_THROW(limited.vals2coefs(fd, fd), std::length_error);
  EXPECT_THROW(limited.convolve(1., fd, gd, out), std::length_error);
  EXPECT_EQ(limited.storage_bytes(), tables);
  out = gpu::Tensor(3, 3, 3);
  limited.coefs2vals(fd, out);
  compare(out, host.coefs2vals(fc));
  out           = gpu::Tensor(3, 3, 2);
  auto capacity = tables + gpu::tensor_bytes(3, 3, 3) + 3 * gpu::tensor_bytes(3, 3, 2);
  gpu::DlrOps bounded(ops, host, capacity);
  bounded.convolve(3., fd, gd, out);
  compare(out, expected);
  EXPECT_EQ(bounded.storage_bytes(), capacity);
  auto big = coefficients(3, 7, 7, .4);
  gpu::Tensor large(big), untouched(big);
  EXPECT_THROW(bounded.convolve(1., large, large, untouched), std::length_error);
  compare(untouched, big);
  EXPECT_EQ(bounded.storage_bytes(), capacity);
  bounded.convolve(3., fd, gd, out);
  compare(out, expected);
  gpu::Tensor wrong(4, 3, 3);
  EXPECT_THROW(device.vals2coefs(wrong, fd), std::invalid_argument);
  EXPECT_THROW(device.reflect(fd, fd), std::invalid_argument);
  EXPECT_THROW(device.convolve(std::numeric_limits<double>::quiet_NaN(), fd, gd, out), std::invalid_argument);
  EXPECT_THROW(device.convolve(1., fd, fd, out), std::invalid_argument);
  device.convolve(3., fd, gd, out);
  compare(out, expected);
  struct SyncGuard {
    bool previous = nda::lapack::device::get_synchronization();
    ~SyncGuard() { nda::lapack::device::set_synchronization(previous); }
  };
  {
    SyncGuard guard;
    nda::lapack::device::set_synchronization(false);
    gpu::Tensor values(host.coefs2vals(fc)), solved(3, 3, 3);
    device.vals2coefs(values, solved);
    compare(solved, host.vals2coefs(host.coefs2vals(fc)));
    EXPECT_FALSE(nda::lapack::device::get_synchronization());
  }
  ops.reset(); // DLR owner retains its stream/handles.
  device.convolve(3., fd, gd, out);
  compare(out, expected);
}
#ifdef NDA_HAVE_CUDA
TEST(BlockSparseCudaDlr, CompactStorageAndAliasTransitions) {
  auto host = small_system();
  auto ops = std::make_shared<gpu::DeviceOps>();
  gpu::DlrOps device(ops, host);
  auto tables = device.storage_bytes();
  auto fc = coefficients(3, 7, 7, .3), gc = coefficients(3, 7, 7, .8);
  gpu::Tensor fv(host.coefs2vals(fc)), gv(host.coefs2vals(gc)), fd(fc.shape()), gd(gc.shape()), out(fc.shape());
  auto tile = gpu::tensor_bytes(3, 7, 7);
  device.vals2coefs(fv, fd);
  device.vals2coefs(gv, gd);
  auto recovered_f = host.vals2coefs(host.coefs2vals(fc));
  auto recovered_g = host.vals2coefs(host.coefs2vals(gc));
  device.convolve(2., fd, gd, out);
  compare(out, host.convolve(2., recovered_f, recovered_g, cppdlr::TIME_ORDERED));
  EXPECT_EQ(device.storage_bytes(), tables + 4 * tile);
  device.convolve(2., fd, gd, fd);
  compare(fd, host.convolve(2., recovered_f, recovered_g, cppdlr::TIME_ORDERED));
  EXPECT_EQ(device.storage_bytes(), tables + 5 * tile);
  fd = recovered_f;
  device.convolve(2., fd, gd, out);
  EXPECT_EQ(device.storage_bytes(), tables + 4 * tile);
  device.vals2coefs(fv, fd);
  EXPECT_EQ(device.storage_bytes(), tables + 5 * tile);
  device.release_scratch();
  EXPECT_EQ(device.storage_bytes(), tables);
}
#endif
#else
TEST(BlockSparseCudaDlr, DISABLED_RequiresCuda) {}
#endif

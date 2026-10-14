#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/memory.hpp>

namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using nda::dcomplex;
  nda::array<dcomplex, 3> function(long r, long m, long n) {
    nda::array<dcomplex, 3> a(r, m, n);
    for (long t = 0; t < r; ++t)
      for (long i = 0; i < m; ++i)
        for (long j = 0; j < n; ++j) a(t, i, j) = dcomplex{.13 + .07 * t - .02 * i + .04 * j, .03 * t + .11 * i - .05 * j};
    return a;
  }
  void compare(gpu::TensorConstView actual, nda::array_const_view<dcomplex, 3> expected) {
    auto host = nda::to_host(actual);
    ASSERT_EQ(host.shape(), expected.shape());
    EXPECT_LE(nda::max_element(nda::abs(host - expected)), 3e-14);
  }
} // namespace
TEST(BlockSparseCudaOps, RectangularSparseProductsAndSnapshot) {
  auto ops = std::make_shared<gpu::DeviceOps>();
  gpu::SparseBlock host{3, 5, {0, 2, 3, 5}, {0, 4, 2, 1, 4}, {{1, .3}, {-.2, .5}, {.4, -.7}, {-.8, .1}, {.6, .2}}};
  auto gold = host;
  gpu::DeviceSparseBlock op(ops, host);
  host.values[0] = 0; // Uploaded operator is an owned snapshot.
  EXPECT_EQ(op.rows(), 3);
  EXPECT_EQ(op.columns(), 5);
  for (long rank : {1, 7})
    for (long anchor : {1, 4, 9}) {
      auto x = function(rank, 5, anchor);
      gpu::Tensor in(x), out(rank, 3, anchor);
      op.apply_left(in, out);
      compare(out, gold.apply_left(x));
      auto first = nda::to_host(out);
      op.apply_left(in, out);
      EXPECT_EQ(nda::max_element(nda::abs(first - nda::to_host(out))), 0.);
      auto y = function(rank, anchor, 3);
      gpu::Tensor yin(y), yout(rank, anchor, 5);
      op.apply_right(yin, yout);
      compare(yout, gold.apply_right(y));
      auto reused = op.workspace_bytes();
      op.apply_right(yin, yout);
      EXPECT_EQ(op.workspace_bytes(), reused);
    }
  EXPECT_LT(op.storage_bytes(), 32ULL * 1024 * 1024);
}
TEST(BlockSparseCudaOps, ScalarZeroBudgetsAndRecovery) {
  auto ops = std::make_shared<gpu::DeviceOps>();
  gpu::SparseBlock scalar{1, 1, {0, 1}, {0}, {{.3, .7}}}, zero{2, 3, {0, 0, 0}, {}, {}};
  EXPECT_THROW(gpu::DeviceSparseBlock(ops, scalar, 0), std::length_error);
  EXPECT_THROW(gpu::DeviceSparseBlock(nullptr, scalar), std::invalid_argument);
  gpu::DeviceSparseBlock op(ops, scalar), empty(ops, zero);
  auto x = function(4, 1, 1);
  gpu::Tensor in(x), out(4, 1, 1);
  EXPECT_THROW(op.apply_left(in, in), std::invalid_argument);
  gpu::Tensor wrong(4, 2, 1);
  EXPECT_THROW(op.apply_left(in, wrong), std::invalid_argument);
  op.apply_left(in, out);
  compare(out, scalar.apply_left(x));
  auto z = function(3, 3, 4);
  gpu::Tensor zin(z), zout(3, 2, 4);
  empty.apply_left(zin, zout);
  compare(zout, zero.apply_left(z));
  auto y = function(3, 4, 2);
  gpu::Tensor yin(y), yout(3, 4, 3);
  empty.apply_right(yin, yout);
  compare(yout, zero.apply_right(y));
  EXPECT_EQ(empty.workspace_bytes(), 0);
  // Budget permits owned CSR only; the nonzero primitive needs library workspace.
  auto base = op.storage_bytes() - op.workspace_bytes();
  gpu::DeviceSparseBlock limited(ops, scalar, base);
  EXPECT_THROW(limited.apply_left(in, out), std::length_error);
  op.apply_left(in, out);
  compare(out, scalar.apply_left(x));
}
TEST(BlockSparseCudaOps, RectangularDenseAndRankProducts) {
  auto ops = std::make_shared<gpu::DeviceOps>();
  auto a = function(5, 3, 7), b = function(5, 7, 2);
  gpu::Tensor da(a), db(b), out(5, 3, 2);
  auto expected = nda::zeros<dcomplex>(5, 3, 2);
  for (long t = 0; t < 5; ++t)
    for (long i = 0; i < 3; ++i)
      for (long j = 0; j < 2; ++j)
        for (long k = 0; k < 7; ++k) expected(t, i, j) += a(t, i, k) * b(t, k, j);
  ops->product(da, db, out);
  compare(out, expected);
  nda::matrix<dcomplex> rank(5, 5);
  for (long i = 0; i < 5; ++i)
    for (long j = 0; j < 5; ++j) rank(i, j) = {.1 + .03 * i - .07 * j, .02 * i + .04 * j};
  nda::cumatrix<dcomplex> drank(rank);
  gpu::Tensor result(5, 3, 7);
  auto gold = nda::zeros<dcomplex>(5, 3, 7);
  for (long t = 0; t < 5; ++t)
    for (long s = 0; s < 5; ++s)
      for (long i = 0; i < 3; ++i)
        for (long j = 0; j < 7; ++j) gold(t, i, j) += rank(t, s) * a(s, i, j);
  ops->rank_product(drank, da, result);
  compare(result, gold);
  EXPECT_THROW(ops->rank_product(drank, da, da), std::invalid_argument);
  ops->copy(da, result);
  compare(result, a);
  ops->axpy(dcomplex{.4, -.3}, da, result);
  compare(result, nda::array<dcomplex, 3>(dcomplex{1.4, -.3} * a));
  ops->zero(result);
  compare(result, nda::zeros<dcomplex>(5, 3, 7));
  nda::vector<dcomplex> weights(5);
  for (long t = 0; t < 5; ++t) weights(t) = {.2 + .03 * t, -.4 + .01 * t};
  nda::cuvector<dcomplex> device_weights(weights);
  auto weighted = a;
  for (long t = 0; t < 5; ++t)
    for (long i = 0; i < 3; ++i)
      for (long j = 0; j < 7; ++j) weighted(t, i, j) *= weights(t);
  ops->weight_rows(da, device_weights, result);
  compare(result, weighted);
  ops->weight_rows(da, device_weights, da);
  compare(da, weighted);
  da = a;
  EXPECT_THROW(ops->product(da, db, result), std::invalid_argument);
  // Error handling leaves the owned stream usable.
  ops->product(da, db, out);
  compare(out, expected);
}
TEST(BlockSparseCudaOps, OwningDeviceAndLifetime) {
  int original = -1, count = 0;
  ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  auto ops = std::make_shared<gpu::DeviceOps>();
  auto op  = std::make_unique<gpu::DeviceSparseBlock>(ops, gpu::SparseBlock{1, 1, {0, 1}, {0}, {{1, 0}}});
  EXPECT_EQ(ops->device(), original);
  ops.reset(); // Snapshot retains its execution context.
  auto x = function(2, 1, 1);
  gpu::Tensor in(x), out(2, 1, 1);
  op->apply_left(in, out);
  compare(out, x);
  if (count > 1) {
    int other = (original + 1) % count;
    ASSERT_EQ(cudaSetDevice(other), cudaSuccess);
    EXPECT_THROW(op->apply_left(in, out), std::invalid_argument);
    op.reset(); // Device storage and handles clean up on their owning device.
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, other);
    ASSERT_EQ(cudaSetDevice(original), cudaSuccess);
  } else
    op.reset();
}
TEST(BlockSparseCudaOps, ComplexTracesAndStridedAccumulation) {
  gpu::DeviceOps ops;
  auto values = function(5, 3, 3);
  gpu::Tensor input(values), traced(5, 1, 1), output(5, 2, 3);
  auto gold = nda::zeros<dcomplex>(5, 1, 1);
  for (long t = 0; t < 5; ++t)
    for (long i = 0; i < 3; ++i) gold(t, 0, 0) += values(t, i, i);
  ops.traces(input, traced);
  compare(traced, gold);
  ops.zero(output);
  ops.add_trace(traced, output, 1, 2);
  ops.add_trace(traced, output, 1, 2);
  auto expected = nda::zeros<dcomplex>(5, 2, 3);
  for (long t = 0; t < 5; ++t) expected(t, 1, 2) = 2. * gold(t, 0, 0);
  compare(output, expected);
  ops.scale(dcomplex{.3, -.7}, traced);
  compare(traced, nda::array<dcomplex, 3>(dcomplex{.3, -.7} * gold));
  EXPECT_THROW(ops.traces(input, input), std::invalid_argument);
  EXPECT_THROW(ops.add_trace(traced, output, 2, 0), std::invalid_argument);
  ops.traces(input, traced);
  compare(traced, gold);
}
#else
TEST(BlockSparseCudaOps, DISABLED_RequiresCuda) {}
#endif

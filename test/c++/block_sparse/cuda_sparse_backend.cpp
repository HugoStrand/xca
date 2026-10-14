#include <gtest/gtest.h>
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#ifdef NDA_HAVE_CUDA
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
}
TEST(BlockSparseBackend, CapabilitySelectionAndBudget) {
  auto ops = std::make_shared<gpu::DeviceOps>();
  gpu::SparseBlock host{2, 3, {0, 2, 3}, {0, 2, 1}, {{.3, .2}, {-.7, .1}, {.9, -.4}}};
  EXPECT_THROW(gpu::DeviceSparseBlock(ops, host, 1024, static_cast<gpu::SparseBackend>(99)), std::invalid_argument);
  if (!gpu::ordered_sparse_available()) {
    EXPECT_THROW(gpu::DeviceSparseBlock(ops, host, 1024, gpu::SparseBackend::Ordered), std::runtime_error);
    return;
  }
  EXPECT_THROW(gpu::DeviceSparseBlock(ops, host, 199, gpu::SparseBackend::Ordered), std::length_error);
  gpu::DeviceSparseBlock native(ops, host, 200, gpu::SparseBackend::Ordered);
  EXPECT_EQ(native.storage_bytes(), 200);
  auto input = nda::ones<nda::dcomplex>(3, 3, 5);
  gpu::Tensor values(input), out(3, 2, 5);
  native.apply_left(values, out);
  EXPECT_LE(nda::max_element(nda::abs(nda::to_host(out) - host.apply_left(input))), 1e-15);
  EXPECT_EQ(native.workspace_bytes(), 0);
  gpu::Tensor wrong(3, 2, 4);
  EXPECT_THROW(native.apply_left(values, wrong), std::invalid_argument);
  gpu::SparseBlock zero{2, 3, {0, 0, 0}, {}, {}};
  gpu::DeviceSparseBlock empty(ops, zero, 1024, gpu::SparseBackend::Ordered);
  empty.apply_left(values, out);
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(out))), 0.);
}
#else
TEST(BlockSparseBackend, DISABLED_RequiresCuda) {}
#endif

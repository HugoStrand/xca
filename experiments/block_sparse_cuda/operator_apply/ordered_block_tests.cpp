// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "ordered_block.hpp"
#include <gtest/gtest.h>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using block_sparse_operator_experiment::OrderedSparseBlock;
  gpu::SparseBlock block() { return {2, 3, {0, 2, 3}, {0, 2, 1}, {{.3, .2}, {-.7, .1}, {.9, -.4}}}; }
} // namespace
TEST(OrderedSparseSnapshot, BudgetShapesAliasesAndZeroRows) {
  auto owner = std::make_shared<gpu::DeviceOps>();
  auto host  = block();
  OrderedSparseBlock snapshot(owner, host, 200);
  EXPECT_EQ(snapshot.storage_bytes(), 200);
  EXPECT_EQ(snapshot.workspace_bytes(), 0);
  EXPECT_THROW(OrderedSparseBlock(owner, host, 199), std::length_error);
  gpu::Tensor input(3, 3, 4), out(3, 2, 4), wrong(3, 2, 5);
  owner->zero(input);
  owner->finish();
  EXPECT_THROW(snapshot.apply_left(input, wrong), std::invalid_argument);
  snapshot.apply_left(input, out);
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(out))), 0.);
  gpu::SparseBlock identity{3, 3, {0, 1, 2, 3}, {0, 1, 2}, {{1., 0.}, {1., 0.}, {1., 0.}}};
  OrderedSparseBlock same(owner, identity, 1024);
  EXPECT_THROW(same.apply_left(input, input), std::invalid_argument);
  gpu::SparseBlock zero{2, 3, {0, 0, 0}, {}, {}};
  OrderedSparseBlock empty(owner, zero, 1024);
  empty.apply_left(input, out);
  gpu::Tensor right_input(3, 4, 2), right_out(3, 4, 3);
  owner->zero(right_input);
  owner->finish();
  empty.apply_right(right_input, right_out);
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(right_out))), 0.);
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(out))), 0.);
}
TEST(OrderedSparseSnapshot, OwnsInputMetadataAndDeviceOwner) {
  auto owner                         = std::make_shared<gpu::DeviceOps>();
  std::weak_ptr<gpu::DeviceOps> weak = owner;
  auto host                          = block();
  auto snapshot                      = std::make_unique<OrderedSparseBlock>(owner, host, 1024);
  auto values                        = nda::ones<nda::dcomplex>(3, 3, 4);
  gpu::Tensor input(values), out(3, 2, 4);
  host.values.assign(host.values.size(), 0.);
  owner.reset();
  EXPECT_FALSE(weak.expired());
  snapshot->apply_left(input, out);
  auto result = nda::to_host(out);
  EXPECT_NEAR(std::abs(result(0, 0, 0) - nda::dcomplex{-.4, .3}), 0., 1e-15);
  snapshot.reset();
  EXPECT_TRUE(weak.expired());
}

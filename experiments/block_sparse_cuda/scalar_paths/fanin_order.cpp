// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../scalar_edges/kernels.hpp"
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#include <gtest/gtest.h>
TEST(ScalarFaninOrder, CancellationKeepsSequentialOrderAndExistingOutput) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  auto owner    = std::make_shared<gpu::DeviceOps>();
  nda::array<Complex, 3> host(3, 7, 1);
  for (long t = 0; t < 3; ++t) {
    host(t, 0, 0) = {1e16, 1e16};
    host(t, 2, 0) = {-1e16, -1e16};
    host(t, 4, 0) = {3. + t, 5. - t};
    host(t, 1, 0) = {7., -7.};
    host(t, 3, 0) = {-1e16, 1e16};
    host(t, 5, 0) = {1e16, -1e16};
    host(t, 6, 0) = {2., 3.};
  }
  auto left = nda::zeros<Complex>(3, 1, 1), right = nda::zeros<Complex>(3, 1, 1);
  left()  = Complex{.2, .3};
  right() = Complex{-.4, .7};
  gpu::Tensor packed(host), output_left(left), output_right(right);
  nda::vector<std::uintptr_t> addresses{reinterpret_cast<std::uintptr_t>(output_left.data()), reinterpret_cast<std::uintptr_t>(output_right.data())};
  nda::vector<long> offsets{0, 3, 7}, lanes{0, 2, 4, 1, 3, 5, 6};
  nda::cuvector<std::uintptr_t> device_addresses(addresses);
  nda::cuvector<long> device_offsets(offsets), device_lanes(lanes);
  owner->wait_for_default_uploads();
  owner->finish();
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (long t = 0; t < 3; ++t) {
      for (long j : {0L, 2L, 4L}) left(t, 0, 0) += host(t, j, 0);
      for (long j : {1L, 3L, 5L, 6L}) right(t, 0, 0) += host(t, j, 0);
    }
    ASSERT_EQ(scalar_edge_experiment::ordered_fanin(packed.data(), device_addresses.data(), device_offsets.data(), device_lanes.data(), 3, 7, 2,
                                                    owner->stream()),
              cudaSuccess);
    owner->finish();
    EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(output_left) - left)), 0.);
    EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(output_right) - right)), 0.);
  }
}

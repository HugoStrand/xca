// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include "../scalar_edges/kernels.hpp"
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#include <gtest/gtest.h>
TEST(ScalarConstants, BroadcastMatchesExpandedWeightsWithTailsAndUpdates) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  auto owner = std::make_shared<gpu::DeviceOps>();
  for (auto [rank, lanes] : {std::pair{3L, 17L}, std::pair{23L, 512L}}) {
    nda::array<Complex, 3> host(rank, lanes, 1), expanded(rank, lanes, 1);
    for (long t = 0; t < rank; ++t)
      for (long lane = 0; lane < lanes; ++lane) host(t, lane, 0) = {.1 * (t + 1), -.03 * (lane + 1)};
    gpu::Tensor broadcast(host), control(host);
    for (int repeat = 0; repeat < 2; ++repeat) {
      nda::vector<Complex> coefficients(lanes);
      for (long lane = 0; lane < lanes; ++lane) {
        coefficients(lane) = lane % 7 == 0 ? Complex{0.} : Complex{.3 * (repeat + 1), .07 * (lane + 1)};
        for (long t = 0; t < rank; ++t) expanded(t, lane, 0) = coefficients(lane);
      }
      nda::cuvector<Complex> constants(coefficients);
      gpu::Tensor weights(expanded);
      owner->wait_for_default_uploads();
      ASSERT_EQ(scalar_edge_experiment::multiply(control.data(), weights.data(), control.data(), control.size(), owner->stream()), cudaSuccess);
      ASSERT_EQ(scalar_edge_experiment::multiply_by_lane(broadcast.data(), constants.data(), broadcast.data(), broadcast.size(), lanes,
                                                        owner->stream()), cudaSuccess);
      owner->finish();
      EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(broadcast) - nda::to_host(control))), 0.);
    }
  }
}

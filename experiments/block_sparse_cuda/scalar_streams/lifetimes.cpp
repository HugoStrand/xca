#include "../scalar_edges/scalar_pack.hpp"
#include "../scalar_resident/resident_kernels.hpp"
#include <gtest/gtest.h>

namespace {
using DeviceOps = triqs_xca::block_sparse::cuda::DeviceOps;
using Tensor = triqs_xca::block_sparse::cuda::Tensor;
using Complex = nda::dcomplex;
#include "../scalar_staging/buffers.inc"
}

TEST(ScalarStreams, RepeatedSourcesAndUnwindingCompleteReaders) {
  constexpr long rank = 3, lanes = 7, rounds = 32;
  auto ops = std::make_shared<DeviceOps>();
  auto input = nda::ones<Complex>(rank, lanes, 1);
  std::vector<Tensor> targets, left, right;
  for (long i = 0; i < rounds; ++i) {
    targets.emplace_back(input);
    left.emplace_back(1, lanes, 1);
    right.emplace_back(1, lanes, 1);
  }
  ops->wait_for_default_uploads();
  ops->finish();
  EXPECT_THROW(([&] {
    ScalarStaging staging(ops, lanes);
    for (long i = 0; i < rounds; ++i) {
      staging.prepare(staging.host_weights);
      for (long lane = 0; lane < lanes; ++lane)
        staging.host_weights(lane) = {0, 1, (i + lane) % 2 ? -1 : 1};
      staging.upload(staging.device_weights, staging.host_weights);
      scalar_edge_experiment::check(scalar_resident_experiment::apply(targets[i].data(), staging.device_weights.data(), rank, lanes, ops->stream()));
      staging.prepare(staging.host_constants);
      for (long lane = 0; lane < lanes; ++lane) staging.host_constants(lane) = Complex{.5 + .01 * i, .02 * lane};
      staging.upload(staging.device_constants, staging.host_constants);
      scalar_edge_experiment::check(scalar_edge_experiment::multiply_by_lane(targets[i].data(), staging.device_constants.data(), targets[i].data(), rank * lanes, lanes, ops->stream()));
      staging.prepare(staging.host_pointers);
      for (long lane = 0; lane < lanes; ++lane)
        staging.host_pointers(lane) = reinterpret_cast<std::uintptr_t>(targets[i].data() + lane);
      // gather expects each pointer to address a contiguous rank vector. This
      // test uses repeated rank=1 gathers, retaining all lane readers in flight.
      staging.upload(staging.device_pointers, staging.host_pointers);
      scalar_edge_experiment::check(scalar_edge_experiment::gather(staging.device_pointers.data(), staging.device_pointers.data(), left[i].data(), right[i].data(), 1, lanes, ops->stream()));
    }
    throw std::runtime_error("exercise staging unwind with queued readers");
  }()), std::runtime_error);
  for (long i = 0; i < rounds; ++i) {
    auto a = nda::to_host(targets[i]), b = nda::to_host(left[i]), c = nda::to_host(right[i]);
    for (long lane = 0; lane < lanes; ++lane) {
      Complex expected = Complex{double((i + lane) % 2 ? -1 : 1)} * Complex{.5 + .01 * i, .02 * lane};
      for (long t = 0; t < rank; ++t) EXPECT_EQ(a(t, lane, 0), expected);
      EXPECT_EQ(b(0, lane, 0), expected);
      EXPECT_EQ(c(0, lane, 0), expected);
    }
  }
}

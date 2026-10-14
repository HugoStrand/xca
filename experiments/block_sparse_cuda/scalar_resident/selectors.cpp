#include "resident_kernels.hpp"
#include "../scalar_edges/kernels.hpp"
#include <triqs_xca/block_sparse/cuda/device_ops.hpp>
#include <gtest/gtest.h>

TEST(ResidentWeights, ExpandedControlStridesIdentityZeroAndUpdates) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using Complex = nda::dcomplex;
  constexpr long rank = 3, lanes = 7;
  auto ops = std::make_shared<gpu::DeviceOps>();
  nda::matrix<Complex> host_table(rank, 3);
  nda::array<Complex, 3> input(rank, lanes, 1);
  for (long t = 0; t < rank; ++t)
    for (long j = 0; j < 3; ++j) host_table(t, j) = {.13 + .02 * t + .03 * j, -.07 + .01 * t - .02 * j};
  for (long i = 0; i < input.size(); ++i) input.data()[i] = {.2 + .01 * i, .1 - .02 * i};
  nda::cumatrix<Complex> table(host_table);
  for (int update = 0; update < 2; ++update) {
    if (update) {
      for (auto &z : host_table) z *= Complex{.7, .1};
      ASSERT_EQ(cudaMemcpyAsync(table.data(), host_table.data(), host_table.size() * sizeof(Complex), cudaMemcpyHostToDevice, ops->stream()), cudaSuccess);
    }
    nda::vector<scalar_resident_experiment::Weight> descriptors(lanes);
    auto expanded = nda::zeros<Complex>(rank, lanes, 1);
    for (long lane = 0; lane < lanes; ++lane) {
      int sign = lane == 4 ? 0 : (lane % 2 ? -1 : 1);
      bool lookup = lane != 0 && lane != 4;
      descriptors(lane) = {lookup ? reinterpret_cast<std::uintptr_t>(table.data() + lane % 3) : 0, 3, sign};
      for (long t = 0; t < rank; ++t) expanded(t, lane, 0) = lookup ? Complex{double(sign)} * host_table(t, lane % 3) : Complex{double(sign)};
    }
    nda::cuvector<scalar_resident_experiment::Weight> device(descriptors);
    gpu::Tensor actual(input), reference(input), weights(expanded);
    ops->wait_for_default_uploads();
    ASSERT_EQ(scalar_resident_experiment::apply(actual.data(), device.data(), rank, lanes, ops->stream()), cudaSuccess);
    ASSERT_EQ(scalar_edge_experiment::multiply(reference.data(), weights.data(), reference.data(), reference.size(), ops->stream()), cudaSuccess);
    ops->finish();
    auto a = nda::to_host(actual), b = nda::to_host(reference);
    EXPECT_EQ(nda::max_element(nda::abs(a - b)), 0.);
  }
}

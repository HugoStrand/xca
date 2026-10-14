// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include "../../../benchmarks/block_sparse_common/fixture.hpp"
#include <gtest/gtest.h>
namespace gpu = triqs_xca::block_sparse::cuda;

TEST(BlockSparseMemoryPlan, NumberSectorsAndPlacementWithoutDevice) {
  for (int modes : {10, 14}) {
    auto dimensions = block_sparse_bench::analytic_dimensions(modes, "number");
    EXPECT_EQ(dimensions.size(), modes + 1);
    EXPECT_EQ(*std::max_element(dimensions.begin(), dimensions.end()), modes == 10 ? 252 : 3432);
    std::size_t states = 0;
    for (long dim : dimensions) states += dim;
    EXPECT_EQ(states, 1ULL << modes);
    for (long rank : {26, 64}) {
      auto resident = gpu::reference_memory_plan(rank, dimensions, 2, 32ULL << 20, modes, false, 3, 2);
      auto streamed = gpu::reference_memory_plan(rank, dimensions, 2, 32ULL << 20, modes, true, 3, 2);
      EXPECT_EQ(resident.device_convolution - resident.device_nca, 6 * resident.maximum_rank_buffer);
      EXPECT_EQ(resident.device_convolution - streamed.device_convolution, resident.function - resident.maximum_rank_buffer);
      EXPECT_EQ(resident.device_nca - streamed.device_nca, resident.function - resident.maximum_rank_buffer);
      EXPECT_EQ(resident.device_correlator, streamed.device_correlator);
      EXPECT_EQ(resident.host_output, resident.function);
      // Sum_q C(N,q)^2 = C(2N,N), independently fixing the full output payload.
      EXPECT_EQ(resident.function, 16ULL * rank * (modes == 10 ? 184756ULL : 40116600ULL));
      if (modes == 14 && rank == 64) {
        EXPECT_GT(streamed.device_convolution, 120ULL << 30);
        EXPECT_LT(streamed.device_nca, 120ULL << 30);
        EXPECT_GT(resident.device_nca, 120ULL << 30);
      }
    }
  }
}
TEST(BlockSparseMemoryPlan, InvalidAndOverflowingExtents) {
  EXPECT_THROW(gpu::reference_memory_plan(0, {1}, 1, 1), std::invalid_argument);
  EXPECT_THROW(gpu::reference_memory_plan(1, {}, 1, 1), std::invalid_argument);
  EXPECT_THROW(gpu::reference_memory_plan(1, {0}, 1, 1), std::invalid_argument);
  EXPECT_THROW(gpu::reference_memory_plan(1, {1}, 0, 1), std::invalid_argument);
  EXPECT_THROW(gpu::reference_memory_plan(1, {1}, 1, 0), std::invalid_argument);
  EXPECT_THROW(gpu::reference_memory_plan(1, {1}, 1, 1, -1), std::invalid_argument);
  EXPECT_THROW(gpu::reference_memory_plan(1, {1}, 1, 1, 0, false, -1, 1), std::invalid_argument);
  EXPECT_THROW(gpu::reference_memory_plan(1, {std::numeric_limits<long>::max()}, 1, 1), std::overflow_error);
  EXPECT_THROW(gpu::reference_memory_plan(1, {50000}, 1, 1), std::length_error);
}
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
TEST(BlockSparseMemoryPlan, MatchesProductionEvaluatorBounds) {
  auto ad = block_sparse_bench::atom(4, "number");
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  constexpr std::size_t operator_budget = 32ULL << 20;
  for (auto special : {gpu::SpecialBackend::Reference, gpu::SpecialBackend::Mapped}) {
    if (special == gpu::SpecialBackend::Mapped && !gpu::mapped_special_available()) continue;
    long width = 0;
    if (special == gpu::SpecialBackend::Mapped)
      for (auto const &set : model.sets()) if (set.size() > 1) width = std::max(width, long(set.size()));
    for (auto placement : {gpu::OutputPlacement::Resident, gpu::OutputPlacement::HostSectors}) {
      auto plan = gpu::reference_memory_plan(mesh.dlr_it().rank(), model.dimensions(), poles.size(), operator_budget, width,
                                             placement == gpu::OutputPlacement::HostSectors, 3, 2);
      gpu::DiagramEvaluator evaluator(model, poles, mesh, 1ULL << 30, operator_budget, gpu::SparseBackend::Library,
                                      special, placement, plan.host_output);
      EXPECT_EQ(evaluator.self_energy_storage_bound(nca), plan.device_nca);
      EXPECT_EQ(evaluator.self_energy_storage_bound(oca), plan.device_convolution);
      EXPECT_EQ(evaluator.self_energy_storage_bound(), plan.device_convolution);
      EXPECT_EQ(evaluator.correlator_storage_bound(3, 2), plan.device_correlator);
    }
  }
}
#endif

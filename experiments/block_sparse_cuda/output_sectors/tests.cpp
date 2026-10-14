#include <gtest/gtest.h>
#include "output_evaluator.hpp"
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
TEST(BlockSparseOutputSectors, CompleteOutputAdmissionAndRecovery) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  for (auto partition : {"number", "mixed", "occupation"}) {
    auto ad           = block_sparse_bench::atom(4, partition);
    auto coefficients = block_sparse_bench::coefficients(ad);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    auto model = gpu::export_model(ad, coefficients);
    auto bytes = gpu::function_bytes(mesh.dlr_it().rank(), model.dimensions());
    auto tile  = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
    gpu::DiagramEvaluator reference(model, poles, mesh);
    gpu::OutputSectorEvaluator streamed(model, poles, mesh);
    nda::array<int, 2> topology{{0, 3}, {1, 4}, {2, 5}};
    EXPECT_EQ(reference.self_energy_storage_bound(topology) - streamed.self_energy_storage_bound(topology), bytes - tile);
    auto device_budget = streamed.self_energy_storage_bound(topology);
    gpu::DiagramEvaluator rejected(model, poles, mesh, device_budget);
    EXPECT_THROW(rejected.compute_self_energy_range(G, topology, 0, 3), std::length_error);
    gpu::OutputSectorEvaluator tight(model, poles, mesh, device_budget);
    auto tight_output    = tight.compute_self_energy_range(G, topology, 0, 3);
    auto resident_output = reference.compute_self_energy_range(G, topology, 0, 3);
    for (long b = 0; b < G.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(tight_output[b].data() - resident_output[b].data())), 0.);
    streamed.set_host_output_budget(bytes - 1);
    EXPECT_THROW(streamed.compute_self_energy_range(G, topology, 0, 3), std::length_error);
    streamed.set_host_output_budget(bytes);
    for (int update = 0; update < 3; ++update) {
      if (update == 1)
        for (long b = 0; b < G.size(); ++b) G[b].data() *= 1.01;
      if (update == 2)
        for (long b = 0; b < G.size(); ++b)
          if (b % 2 == 0) G[b].data() = 0.;
      auto expected = reference.compute_self_energy_range(G, topology, 0, 3);
      auto actual   = streamed.compute_self_energy_range(G, topology, 0, 3);
      ASSERT_EQ(actual.size(), expected.size());
      for (long b = 0; b < G.size(); ++b) {
        for (auto z : actual[b].data()) ASSERT_TRUE(std::isfinite(z.real()) && std::isfinite(z.imag()));
        EXPECT_EQ(nda::max_element(nda::abs(actual[b].data() - expected[b].data())), 0.);
      }
      EXPECT_EQ(streamed.last_contributing_paths(), reference.last_contributing_paths());
    }
    auto empty = streamed.compute_self_energy_range(G, topology, 2, 2);
    for (long b = 0; b < empty.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(empty[b].data())), 0.);
  }
}

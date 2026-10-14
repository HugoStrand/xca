// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/cuda/operator_column_maps.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <iostream>

TEST(BlockSparseNcaLiveness, TightAdmissionDynamicFallbacksAndChangedInputs) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  using triqs::operators::c;
  using triqs::operators::c_dag;
  for (auto partition : {"number", "mixed", "occupation"}) {
    auto ad = block_sparse_bench::atom(4, partition);
    auto coefficients = block_sparse_bench::coefficients(ad);
    many_body_operator_complex Q{.01}, M{.03};
    for (int i = 0; i < 4; ++i) Q += n("f", i);
    if (std::string(partition) == "occupation") M += n("f", 0) + .7 * n("f", 1);
    else M += c_dag("f", 0) * c("f", 1) + .7 * c_dag("f", 2) * c("f", 3);
    std::vector<many_body_operator_complex> interactions{Q, M};
    auto dynamic = nda::zeros<nda::dcomplex>(2, 2, 2);
    for (long p = 0; p < 2; ++p) {
      dynamic(p, 0, 0) = .03 - .01 * p;
      dynamic(p, 1, 1) = .02 + .005 * p;
    }
    auto model = gpu::export_model(ad, coefficients, interactions, dynamic);
    bool nonmapping = false;
    for (auto const &set : model.sets()) if (set.size() > 1)
      for (long b = 0; b < long(model.dimensions().size()); ++b) {
        std::vector<gpu::SparseBlock const *> blocks;
        for (int field : set) blocks.push_back(&model.field(field).blocks[b]);
        if (blocks.front()->rows > 0 && !gpu::make_operator_column_maps(blocks)) nonmapping = true;
      }
    if (std::string(partition) != "occupation") ASSERT_TRUE(nonmapping);
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    nda::array<int, 2> topology{{0, 1}}, oca{{0, 2}, {1, 3}};
    auto tile = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
    for (auto sparse : {gpu::SparseBackend::Library, gpu::SparseBackend::Ordered}) {
      if (sparse == gpu::SparseBackend::Ordered && !gpu::ordered_sparse_available()) continue;
      for (auto special : {gpu::SpecialBackend::Reference, gpu::SpecialBackend::Mapped}) {
        if (special == gpu::SpecialBackend::Mapped && !gpu::mapped_special_available()) continue;
        for (auto placement : {gpu::OutputPlacement::Resident, gpu::OutputPlacement::HostSectors}) {
          SCOPED_TRACE(partition);
          gpu::DiagramEvaluator reference(model, poles, mesh, 1ULL << 30, 32ULL << 20, sparse, special, placement);
          auto bound = reference.self_energy_storage_bound(topology);
          EXPECT_EQ(reference.self_energy_storage_bound() - bound, 6 * tile);
          gpu::DiagramEvaluator rejected(model, poles, mesh, bound - 1, 32ULL << 20, sparse, special, placement);
          EXPECT_THROW(rejected.compute_self_energy(G, topology), std::length_error);
          EXPECT_EQ(rejected.last_device_peak_delta(), 0);
          gpu::DiagramEvaluator tight(model, poles, mesh, bound, 32ULL << 20, sparse, special, placement);
          for (int update = 0; update < 2; ++update) {
            if (update) for (long b = 0; b < G.size(); ++b) G[b].data() *= .97;
            auto expected = reference.compute_self_energy(G, topology);
            auto actual = tight.compute_self_energy(G, topology);
            double signal = 0;
            for (long b = 0; b < G.size(); ++b) {
              EXPECT_EQ(nda::max_element(nda::abs(actual[b].data() - expected[b].data())), 0.);
              signal = std::max(signal, nda::max_element(nda::abs(actual[b].data())));
            }
            EXPECT_GT(signal, 1e-20);
            EXPECT_GT(tight.last_contributing_paths(), 0);
            EXPECT_EQ(tight.last_contributing_paths(), reference.last_contributing_paths());
          }
          auto peak = tight.last_device_peak_delta();
          EXPECT_LE(peak, bound + (64ULL << 20));
          EXPECT_THROW(tight.compute_self_energy_range(G, oca, 0, 1), std::length_error);
          std::cout << "nca-live," << partition << ',' << int(sparse) << ',' << int(special) << ',' << int(placement)
                    << ',' << bound << ',' << bound + tile << ',' << tile << ',' << peak << '\n';
        }
      }
    }
  }
}
TEST(BlockSparseNcaLiveness, BoundedTenModeNumberSectorCall) {
  namespace gpu = triqs_xca::block_sparse::cuda;
  auto ad = block_sparse_bench::atom(10, "number");
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> topology{{0, 1}};
  auto tile = gpu::inventory_memory(mesh.dlr_it().rank(), model.dimensions(), 0).maximum_rank_buffer;
  for (auto placement : {gpu::OutputPlacement::Resident, gpu::OutputPlacement::HostSectors}) {
    gpu::DiagramEvaluator reference(model, poles, mesh, 1ULL << 30, 32ULL << 20, gpu::SparseBackend::Library,
                                    gpu::SpecialBackend::Reference, placement);
    auto bound = reference.self_energy_storage_bound(topology);
    gpu::DiagramEvaluator tight(model, poles, mesh, bound, 32ULL << 20, gpu::SparseBackend::Library,
                                gpu::SpecialBackend::Reference, placement);
    auto expected = reference.compute_self_energy_range(G, topology, 0, 1);
    auto actual = tight.compute_self_energy_range(G, topology, 0, 1);
    for (long b = 0; b < G.size(); ++b)
      EXPECT_EQ(nda::max_element(nda::abs(actual[b].data() - expected[b].data())), 0.);
    EXPECT_GT(tight.last_contributing_paths(), 0);
    EXPECT_LE(tight.last_device_peak_delta(), bound + (64ULL << 20));
    std::cout << "nca-live,n10-number,0,0," << int(placement) << ',' << bound << ',' << bound + tile
              << ',' << tile << ',' << tight.last_device_peak_delta() << '\n';
  }
}
#else
TEST(BlockSparseNcaLiveness, DISABLED_RequiresCuda) {}
#endif

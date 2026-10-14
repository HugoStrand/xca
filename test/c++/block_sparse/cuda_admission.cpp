// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/dlr_ops.hpp>
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include <triqs_xca/block_sparse/cuda/memory.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <gtest/gtest.h>
namespace {
  namespace gpu = triqs_xca::block_sparse::cuda;
  void exact(block_sparse_bench::Gf const &a, block_sparse_bench::Gf const &b) {
    ASSERT_EQ(a.size(), b.size());
    for (long block = 0; block < a.size(); ++block) EXPECT_EQ(nda::max_element(nda::abs(a[block].data() - b[block].data())), 0.);
  }
} // namespace
TEST(BlockSparseCudaAdmission, LowerBudgetRejectionAndHigherOrderTransition) {
  auto ad    = block_sparse_bench::atom(4, "number");
  auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
  triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 20., 1e-10, false);
  auto G = block_sparse_bench::propagator(ad, mesh);
  nda::vector<double> poles{-1.5, .6};
  nda::array<int, 2> nca{{0, 1}}, oca{{0, 2}, {1, 3}};
  for (auto special : {gpu::SpecialBackend::Reference, gpu::SpecialBackend::Mapped}) {
    if (special == gpu::SpecialBackend::Mapped && !gpu::mapped_special_available()) continue;
    gpu::DiagramEvaluator candidate(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Library, special);
    auto bound = candidate.self_energy_storage_bound(nca);
    EXPECT_EQ(candidate.self_energy_storage_bound() - bound, 5 * gpu::tensor_bytes(mesh.dlr_it().rank(), 6, 6));
    EXPECT_EQ(candidate.self_energy_storage_bound(oca), candidate.self_energy_storage_bound());
    gpu::DiagramEvaluator limited(model, poles, mesh, bound - 1, 32ULL * 1024 * 1024, gpu::SparseBackend::Library, special);
    EXPECT_THROW(limited.compute_self_energy_range(G, nca, 0, 1), std::length_error);
    EXPECT_EQ(limited.last_path_counts().accepted, 0);
    EXPECT_EQ(limited.last_device_peak_delta(), 0);
    auto empty = limited.compute_self_energy_range(G, nca, 0, 0);
    for (long b = 0; b < empty.size(); ++b) EXPECT_EQ(nda::max_element(nda::abs(empty[b].data())), 0.);
    gpu::DiagramEvaluator tight(model, poles, mesh, bound, 32ULL * 1024 * 1024, gpu::SparseBackend::Library, special);
    gpu::DiagramEvaluator reference(model, poles, mesh, 1024ULL * 1024 * 1024, 32ULL * 1024 * 1024, gpu::SparseBackend::Library, special);
    exact(tight.compute_self_energy(G, nca), reference.compute_self_energy(G, nca));
    EXPECT_THROW(tight.compute_self_energy_range(G, oca, 0, 1), std::length_error);
    exact(tight.compute_self_energy(G, nca), reference.compute_self_energy(G, nca));
    candidate.compute_self_energy_range(G, oca, 0, 1); // Populates retained DLR scratch.
    exact(candidate.compute_self_energy(G, nca), reference.compute_self_energy(G, nca));
    exact(candidate.compute_self_energy_range(G, oca, 0, 1), reference.compute_self_energy_range(G, oca, 0, 1));
  }
}
TEST(BlockSparseCudaAdmission, ScratchReleaseRetainsTablesAndSolvesAgain) {
  cppdlr::imtime_ops host(20., cppdlr::build_dlr_rf(20., 1e-10, false), false);
  auto owner = std::make_shared<gpu::DeviceOps>();
  gpu::DlrOps device(owner, host);
  auto tables = device.storage_bytes();
  long rank   = host.rank();
  auto fc     = nda::zeros<nda::dcomplex>(rank, 3, 3);
  auto gc     = fc;
  for (long t = 0; t < rank; ++t)
    for (long i = 0; i < 3; ++i) {
      fc(t, i, i) = .1 * (t + 1);
      gc(t, i, i) = .03 * (t + 1);
    }
  gpu::Tensor values(host.coefs2vals(fc)), fd(fc.shape()), gd(gc), out(fc.shape());
  for (int repeat = 0; repeat < 2; ++repeat) {
    device.vals2coefs(values, fd);
    device.convolve(2., fd, gd, out);
    auto result = nda::to_host(out);
    EXPECT_GT(device.storage_bytes(), tables);
    device.release_scratch();
    EXPECT_EQ(device.storage_bytes(), tables);
    device.vals2coefs(values, fd);
    device.convolve(2., fd, gd, out);
    EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(out) - result)), 0.);
    device.release_scratch();
    device.release_scratch();
    EXPECT_EQ(device.storage_bytes(), tables);
  }
}

#else
#include <gtest/gtest.h>
TEST(BlockSparseCudaAdmission, DISABLED_RequiresCuda) {}
#endif

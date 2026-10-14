// Copyright (c) 2026, The Simons Foundation
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#ifdef NDA_HAVE_CUDA
#include <triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>
#include "../../../benchmarks/block_sparse_common/call_runner.hpp"
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

namespace gpu = triqs_xca::block_sparse::cuda;
namespace {
  // A device trap poisons its context. Run only in Google's re-exec child and
  // exit without unwinding GPU owners; the OS releases the child's context.
  // Deliberate fault injection is not a sanitizer-clean test.
  [[noreturn]] void fail(int status, char const *message) {
    std::fprintf(stderr, "%s\n", message);
    std::_Exit(status);
  }
  template <typename T> T symbol(void *library, char const *name) {
    auto result = reinterpret_cast<T>(dlsym(library, name));
    if (!result) fail(2, "CUDA driver symbol missing");
    return result;
  }
  [[noreturn]] void submit_and_check() {
    auto ad = block_sparse_bench::atom(2, "number");
    triqs::mesh::dlr_imtime mesh(2., triqs::mesh::Fermion, 10., 1e-10, false);
    auto G = block_sparse_bench::propagator(ad, mesh);
    nda::vector<double> poles{-1.5, .6};
    auto model = gpu::export_model(ad, block_sparse_bench::coefficients(ad));
    nda::array<int, 2> topology{{0, 1}};
    std::vector<gpu::DiagramEvaluator *> evaluators;
    for (auto policy : {gpu::StoragePolicy::Resident, gpu::StoragePolicy::HostInputs, gpu::StoragePolicy::StreamFragments}) {
      auto *e = new gpu::DiagramEvaluator(model, poles, mesh, 1ULL << 30, 32ULL << 20, gpu::SparseBackend::Library, gpu::SpecialBackend::Reference,
                                          gpu::OutputPlacement::Resident, 512ULL << 20, gpu::NcaBackend::Reference, policy);
      auto output = e->compute_self_energy(G, topology);
      if (!e->last_contributing_paths()) fail(3, "Normal evaluator preflight has no work");
      evaluators.push_back(e);
    }
    auto *ops = new gpu::DeviceOps;
    ops->finish();
    void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!driver) fail(2, "CUDA driver unavailable");
    auto load     = symbol<decltype(&cuModuleLoadData)>(driver, "cuModuleLoadData");
    auto function = symbol<decltype(&cuModuleGetFunction)>(driver, "cuModuleGetFunction");
    auto launch   = symbol<decltype(&cuLaunchKernel)>(driver, "cuLaunchKernel");
    constexpr char ptx[] =
       ".version 6.0\n.target sm_50\n.address_size 64\n"
       ".visible .entry deliberate_fault() { trap; ret; }\n";
    CUmodule module{};
    CUfunction kernel{};
    if (load(&module, ptx) != CUDA_SUCCESS || function(&kernel, module, "deliberate_fault") != CUDA_SUCCESS)
      fail(2, "CUDA fault kernel preparation failed");
    if (launch(kernel, 1, 1, 1, 1, 1, 1, 0, ops->stream(), nullptr, nullptr) != CUDA_SUCCESS) fail(4, "Fault was rejected before submission");
    bool propagated = false;
    try {
      ops->finish();
    } catch (std::runtime_error const &) { propagated = true; }
    if (!propagated) fail(5, "Owned stream did not propagate asynchronous fault");
    auto status = cudaDeviceSynchronize();
    if (status != cudaErrorLaunchFailure && status != cudaErrorIllegalInstruction) fail(6, "Unexpected sticky CUDA error");
    for (auto *e : evaluators) {
      bool rejected = false;
      try {
        e->compute_self_energy(G, topology);
      } catch (std::runtime_error const &) { rejected = true; }
      if (!rejected) fail(7, "Evaluator returned a result after device failure");
    }
    fail(0, "asynchronous fault propagated; all policies reject poisoned context");
  }
} // namespace

TEST(BlockSparseCudaFailure, SubmittedFaultPropagatesWithoutFallback) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(submit_and_check(), ::testing::ExitedWithCode(0), "asynchronous fault propagated");
  // Child failure must not poison the parent context or later CTests.
  gpu::DeviceOps ops;
  gpu::Tensor value(2, 1, 1);
  ops.zero(value);
  ops.finish();
  EXPECT_EQ(nda::max_element(nda::abs(nda::to_host(value))), 0.);
}
#else
TEST(BlockSparseCudaFailure, DISABLED_RequiresCuda) {}
#endif

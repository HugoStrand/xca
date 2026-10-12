#include <gtest/gtest.h>
#ifdef XCA_HAVE_CUDA_SMALL
#include <triqs_xca/dense/cuda_small/diagram_evaluator.hpp>
#include "../../../../benchmarks/dense_cuda_small/harness.hpp"

TEST(CudaSmallBatch, OrderedWidthsTailsAndChangedInputs) {
  using namespace dense_cuda_bench;
  for (int N : {4, 8}) {
    Options options;
    options.N = N;
    Fixture f(options);
    triqs_xca::dense::cuda_small::DiagramEvaluator candidate(options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
    ASSERT_EQ(candidate.rank(), 16);
    auto reference = baseline(f);
    auto topology = dense_cuda_bench::topology(6, "crossing");
    long total = candidate.get_num_self_energy_backbones(topology);
    for (int count : {1, 8, 32, 128, 129, 259}) {
      SCOPED_TRACE(::testing::Message() << "N=" << N << " count=" << count);
      auto indices = select_indices(total, count, 1729);
      if (count > 1) indices(count-1) = indices(0);
      auto expected = reference.evaluate("se", topology, indices);
      for (int repeat = 0; repeat < 2; ++repeat) {
        auto value = candidate.compute_self_energy(f.G, topology, indices);
        EXPECT_NO_THROW(compare(tensor(value[0].data()), expected));
      }
    }
    EXPECT_GT(candidate.batch_chunks(), 0);
    auto indices = select_indices(total, 129, 31415);
    f.G[0].data() *= complex{.99, .001};
    auto expected = reference.evaluate("se", topology, indices);
    auto value = candidate.compute_self_energy(f.G, topology, indices);
    EXPECT_NO_THROW(compare(tensor(value[0].data()), expected));
    nda::vector<long> invalid{indices(0), total};
    EXPECT_THROW(candidate.compute_self_energy(f.G, topology, invalid), std::exception);
    value = candidate.compute_self_energy(f.G, topology, indices);
    EXPECT_NO_THROW(compare(tensor(value[0].data()), expected));
    nda::vector<long> empty(0);
    value = candidate.compute_self_energy(f.G, topology, empty);
    EXPECT_EQ(nda::max_element(nda::abs(value[0].data())), 0.);
  }
}

TEST(CudaSmallBatch, CapacityAndWorkspaceFallback) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  triqs_xca::dense::cuda_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  auto reference = baseline(f);
  auto t = topology(3, "chain");
  auto indices = select_indices(candidate.get_num_self_energy_backbones(t), 129, 1729);
  auto expected = reference.evaluate("se", t, indices);
  for (int capacity : {1, 8, 32, 128}) {
    candidate.set_batch_capacity(capacity);
    auto value = candidate.compute_self_energy(f.G, t, indices);
    EXPECT_NO_THROW(compare(tensor(value[0].data()), expected));
  }
  candidate.set_batch_capacity(1);
  candidate.set_workspace_budget(candidate.workspace_bytes());
  candidate.set_batch_capacity(128);
  auto before = candidate.batch_chunks();
  auto value = candidate.compute_self_energy(f.G, t, indices);
  EXPECT_NO_THROW(compare(tensor(value[0].data()), expected));
  EXPECT_EQ(candidate.batch_chunks(), before);
  EXPECT_LE(candidate.workspace_bytes(), candidate.workspace_budget());
  EXPECT_THROW(candidate.set_batch_capacity(0), std::invalid_argument);
  EXPECT_THROW(candidate.set_workspace_budget(0), std::invalid_argument);
}
TEST(CudaSmallBatch, FullEnumerationBeyondPrototypeLimit) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  triqs_xca::dense::cuda_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  auto reference = baseline(f);
  auto t = topology(5, "chain");
  long count = candidate.get_num_self_energy_backbones(t);
  ASSERT_GT(count, 4096);
  nda::vector<long> indices(count);
  for (long i = 0; i < count; ++i) indices(i) = i;
  auto expected = reference.evaluate("se", t, indices);
  auto value = candidate.compute_self_energy(f.G, t);
  EXPECT_NO_THROW(compare(tensor(value[0].data()), expected));
  EXPECT_GT(candidate.batch_chunks(), 0);
}

#else
TEST(CudaSmallBatch, DISABLED_RequiresCudaSmall) {}
#endif

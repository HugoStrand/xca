#include <gtest/gtest.h>
#include <thread>
#ifdef XCA_HAVE_CUDA_SMALL
#include <triqs_xca/dense/cuda_small/diagram_evaluator.hpp>
#include "../../../../benchmarks/dense_cuda_small/harness.hpp"

namespace triqs_xca::dense::cuda_small {
  struct BatchFailureProbe {
    static void arm(DiagramEvaluator &e, bool length_error) {
      e.batches_[2]->schedule_.push_back([length_error] {
        if (length_error) throw std::length_error("injected failure after contribution submission");
        throw std::bad_alloc();
      });
    }
    static void disarm(DiagramEvaluator &e) {
      if (e.batches_[2]) e.batches_[2]->schedule_.pop_back();
    }
  };
}

TEST(CudaSmallBatch, SubmittedFailuresAreNotReplayed) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  triqs_xca::dense::cuda_small::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto reference=baseline(f);
  auto t=topology(3,"crossing");
  auto indices=select_indices(candidate.get_num_self_energy_backbones(t),128,1729);
  auto expected=reference.evaluate("se",t,indices);
  candidate.compute_self_energy(f.G,t,indices); // Populate the content-keyed schedule.
  for (bool length_error : {false,true}) {
    triqs_xca::dense::cuda_small::BatchFailureProbe::arm(candidate,length_error);
    if (length_error) EXPECT_THROW(candidate.compute_self_energy(f.G,t,indices),std::length_error);
    else EXPECT_THROW(candidate.compute_self_energy(f.G,t,indices),std::bad_alloc);
    triqs_xca::dense::cuda_small::BatchFailureProbe::disarm(candidate);
    auto actual=candidate.compute_self_energy(f.G,t,indices);
    EXPECT_NO_THROW(compare(tensor(actual[0].data()),expected));
  }
}

TEST(CudaSmallBatch, StorageBoundsChangedSchedulesAndFallback) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  triqs_xca::dense::cuda_small::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto reference=baseline(f);
  auto t=topology(3,"crossing");
  auto indices=select_indices(candidate.get_num_self_energy_backbones(t),128,1729);
  auto initial_device=candidate.device_workspace_bytes();
  auto initial_host=candidate.host_workspace_bound_bytes();
  auto initial_unified=candidate.unified_workspace_bytes();
  std::size_t free_before=0,total=0,free_after=0;
  nda::device_error_check(cudaMemGetInfo(&free_before,&total),"memory before cache");
  candidate.compute_self_energy(f.G,t,indices);
  nda::device_error_check(cudaMemGetInfo(&free_after,&total),"memory after cache");
  EXPECT_GT(candidate.device_workspace_bytes(),initial_device);
  EXPECT_GT(candidate.host_workspace_bound_bytes(),initial_host);
  EXPECT_GT(candidate.unified_workspace_bytes(),initial_unified);
  // CUDA-library storage and allocation granularity may exceed reported payloads.
  auto observed=free_before>free_after ? free_before-free_after : 0;
  EXPECT_GE(observed+2*1024*1024,candidate.device_workspace_bytes()-initial_device);
  auto budget=candidate.workspace_bytes();
  candidate.set_workspace_budget(budget); // Evict caches; subsequent growth must fit.
  for (int seed : {31415,1729,27182}) {
    indices=select_indices(candidate.get_num_self_energy_backbones(t),129,seed);
    f.G[0].data()*=complex{.99,.001};
    auto actual=candidate.compute_self_energy(f.G,t,indices);
    EXPECT_NO_THROW(compare(tensor(actual[0].data()),reference.evaluate("se",t,indices)));
    EXPECT_LE(candidate.workspace_bytes(),budget);
  }
}

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

TEST(CudaSmallBatch, NativeRangesMatchIndexedWork) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  triqs_xca::dense::cuda_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  auto reference = baseline(f);
  auto t = topology(3, "crossing");
  for (std::string method : {"se", "pairs", "spgf"}) {
    long total = method == "spgf" ? candidate.get_num_single_ptcle_gf_backbones(t) : candidate.get_num_self_energy_backbones(t);
    for (auto bounds : {std::pair<long,long>{0,0}, {0,1}, {0,9}, {7,47}, {total-1,total}, {total,total}}) {
      auto [first,last] = bounds;
      SCOPED_TRACE(::testing::Message() << method << " [" << first << "," << last << ")");
      std::vector<long> selected;
      long group = total / 8;
      for (long i = first; i < last; ++i) if (method != "pairs" || (i / group) % 2 == 0) selected.push_back(i);
      nda::vector<long> indices(selected.size());
      for (long i = 0; i < indices.size(); ++i) indices(i) = selected[i];
      auto expected = reference.evaluate(method, t, indices);
      tensor actual;
      if (method == "se") actual = candidate.compute_self_energy_range(f.G, t, first, last)[0].data();
      else if (method == "pairs") actual = candidate.compute_self_energy_by_pairs_range(f.G, t, first, last)[0].data();
      else actual = candidate.compute_single_ptcle_gf_range(f.G, t, first, last);
      EXPECT_NO_THROW(compare(actual, expected));
    }
  }
  EXPECT_THROW(candidate.compute_self_energy_range(f.G, t, -1, 0), std::out_of_range);
  EXPECT_THROW(candidate.compute_self_energy_range(f.G, t, 10, 9), std::out_of_range);
  EXPECT_THROW(candidate.compute_self_energy_by_pairs_range(f.G, t, 0, candidate.get_num_self_energy_backbones(t)+1), std::out_of_range);
  EXPECT_THROW(candidate.compute_single_ptcle_gf_range(f.G, t, 0, candidate.get_num_single_ptcle_gf_backbones(t)+1), std::out_of_range);
}

TEST(CudaSmallBatch, NativeOneTimeRangesMatchIndexedWork) {
  using namespace dense_cuda_bench;
  using namespace triqs::operators;
  Fixture f(Options{});
  triqs_xca::dense::cuda_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  triqs_xca::dense::cuda::DiagramEvaluator reference(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  triqs::atom_diag::fundamental_operator_set fs;
  fs.insert("b", 0); fs.insert("b", 1);
  many_body_operator_complex H = .1*n("b",0) + .3*n("b",1);
  triqs::atom_diag::atom_diag<true> ad(H, fs, std::vector<many_body_operator_complex>{});
  auto t = topology(3, "crossing");
  for (bool fermionic : {false, true}) {
    std::vector<many_body_operator_complex> tau = fermionic ? std::vector<many_body_operator_complex>{c("b",0),c("b",1)} : std::vector<many_body_operator_complex>{n("b",0)};
    std::vector<many_body_operator_complex> zero = fermionic ? std::vector<many_body_operator_complex>{c_dag("b",0),c_dag("b",1)} : std::vector<many_body_operator_complex>{n("b",1)};
    for (auto bounds : {std::pair<long,long>{0,0}, {0,9}, {7,47}}) {
      auto [first,last] = bounds;
      nda::vector<long> indices(last-first);
      for (long i=0; i<indices.size(); ++i) indices(i)=first+i;
      auto expected = reference.compute_one_time_correlator(f.G,tau,zero,ad,t,indices);
      auto actual = candidate.compute_one_time_correlator_range(f.G,tau,zero,ad,t,first,last);
      EXPECT_NO_THROW(compare(actual,expected));
    }
  }
}

TEST(CudaSmallBatch, PairedBatchesKeepMethodCacheAndRangeOrder) {
  using namespace dense_cuda_bench;
  for (int N : {4,8}) {
    Options options; options.N=N;
    Fixture f(options);
    triqs_xca::dense::cuda_small::DiagramEvaluator candidate(options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
    auto reference=baseline(f);
    auto t=topology(6,"crossing");
    long total=candidate.get_num_self_energy_backbones(t), group=total/64;
    for (int count : {8,32,128,129}) {
      auto ordinal=select_indices(total/2,count,1729);
      nda::vector<long> indices(count);
      for (int i=0; i<count; ++i) indices(i)=2*(ordinal(i)/group)*group+ordinal(i)%group;
      indices(count-1)=indices(0);
      auto expected=reference.evaluate("pairs",t,indices);
      for (int repeat=0; repeat<2; ++repeat) {
        auto ordinary=candidate.compute_self_energy(f.G,t,indices);
        EXPECT_NO_THROW(compare(tensor(ordinary[0].data()),reference.evaluate("se",t,indices)));
        auto before=candidate.batch_chunks();
        auto paired=candidate.compute_self_energy_by_pairs(f.G,t,indices);
        EXPECT_NO_THROW(compare(tensor(paired[0].data()),expected)) << "N=" << N << " count=" << count;
        EXPECT_GT(candidate.batch_chunks(),before);
      }
    }
    long first=group-5,last=2*group+9;
    std::vector<long> selected;
    for (long i=first; i<last; ++i) if ((i/group)%2==0) selected.push_back(i);
    nda::vector<long> indices(selected.size());
    for (long i=0; i<indices.size(); ++i) indices(i)=selected[i];
    auto expected=reference.evaluate("pairs",t,indices);
    auto actual=candidate.compute_self_energy_by_pairs_range(f.G,t,first,last);
    EXPECT_NO_THROW(compare(tensor(actual[0].data()),expected));
  }
}

TEST(CudaSmallBatch, CorrelatorBudgetFailureIsRecoverable) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  triqs_xca::dense::cuda_small::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  triqs_xca::dense::cuda::DiagramEvaluator reference(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto t=topology(2,"crossing");
  triqs_xca::backbone::CorrelatorBackbone b(t,f.n), gold(t,f.n);
  candidate.set_workspace_budget(candidate.workspace_bytes());
  EXPECT_THROW(candidate.eval_correlator(f.G[0].data(),b,f.Fs,f.Fd,0L),std::length_error);
  candidate.set_workspace_budget(512*1024*1024);
  auto expected=reference.eval_correlator(f.G[0].data(),gold,f.Fs,f.Fd,0L);
  auto actual=candidate.eval_correlator(f.G[0].data(),b,f.Fs,f.Fd,0L);
  EXPECT_NO_THROW(compare(actual,expected));
}

TEST(CudaSmallBatch, InterleavedInstancesAndCrossThreadDestruction) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  using Evaluator=triqs_xca::dense::cuda_small::DiagramEvaluator;
  auto first=std::make_unique<Evaluator>(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto second=std::make_unique<Evaluator>(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  first->set_batch_capacity(8);
  second->set_batch_capacity(128);
  auto reference=baseline(f);
  auto t=topology(3,"crossing");
  auto indices=select_indices(first->get_num_self_energy_backbones(t),129,31415);
  for (int repeat=0; repeat<2; ++repeat) {
    auto expected=reference.evaluate("se",t,indices);
    auto actual=first->compute_self_energy(f.G,t,indices);
    EXPECT_NO_THROW(compare(tensor(actual[0].data()),expected));
    auto paired=second->compute_self_energy_by_pairs(f.G,t,indices);
    EXPECT_NO_THROW(compare(tensor(paired[0].data()),reference.evaluate("pairs",t,indices)));
    f.G[0].data()*=complex{.97,.002};
  }
  EXPECT_GT(first->batch_chunks(),0);
  EXPECT_GT(second->batch_chunks(),0);
  std::thread cleanup([owned=std::move(first)]() mutable { owned.reset(); });
  cleanup.join();
  auto expected=reference.evaluate("se",t,indices);
  auto actual=second->compute_self_energy(f.G,t,indices);
  EXPECT_NO_THROW(compare(tensor(actual[0].data()),expected));
}

TEST(CudaSmallBatch, DeviceSwitchRejectsWorkAndCleanupRestoresCaller) {
  using namespace dense_cuda_bench;
  int count=0,owner=0;
  nda::device_error_check(cudaGetDeviceCount(&count),"device count");
  if (count<2) GTEST_SKIP() << "requires two visible CUDA devices";
  nda::device_error_check(cudaGetDevice(&owner),"current device");
  struct Restore { int device; ~Restore() { cudaSetDevice(device); } } restore{owner};
  Fixture f(Options{});
  using Evaluator=triqs_xca::dense::cuda_small::DiagramEvaluator;
  auto candidate=std::make_unique<Evaluator>(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto t=topology(3,"crossing");
  auto indices=select_indices(candidate->get_num_self_energy_backbones(t),128,1729);
  candidate->compute_self_energy(f.G,t,indices); // Include a resident batch owner.
  int other=(owner+1)%count;
  nda::device_error_check(cudaSetDevice(other),"select other device");
  EXPECT_THROW(candidate->compute_self_energy(f.G,t,indices),std::runtime_error);
  EXPECT_THROW(candidate->compute_single_ptcle_gf(f.G,t,indices),std::runtime_error);
  EXPECT_THROW(candidate->set_batch_capacity(32),std::runtime_error);
  candidate.reset();
  int current=-1;
  nda::device_error_check(cudaGetDevice(&current),"device after cleanup");
  EXPECT_EQ(current,other);
}

#else
TEST(CudaSmallBatch, DISABLED_RequiresCudaSmall) {}
#endif

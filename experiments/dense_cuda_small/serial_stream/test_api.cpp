#include <gtest/gtest.h>
#include "../../../benchmarks/dense_cuda_small/harness.hpp"
#include "diagram_evaluator.hpp"
TEST(StreamAPI, InvalidPrefixRecoveryAndLongIndices) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  xca_experiments::serial_stream::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto reference=baseline(f);
  auto t=topology(12,"crossing");
  long total=candidate.get_num_self_energy_backbones(t);
  auto ix=select_indices(total,3,1729);
  ASSERT_GT(total,long(INT_MAX)+17);
  ix(0)=long(INT_MAX)+17;
  nda::vector<long> bad{ix(0),total};
  EXPECT_THROW(candidate.compute_self_energy(f.G,t,bad),std::exception);
  auto value=candidate.compute_self_energy(f.G,t,ix);
  EXPECT_NO_THROW(compare(tensor(value[0].data()),reference.evaluate("se",t,ix)));
  nda::vector<long> empty(0);
  auto zero=candidate.compute_self_energy(f.G,t,empty);
  EXPECT_EQ(nda::max_element(nda::abs(zero[0].data())),0.);
  EXPECT_THROW(candidate.compute_self_energy_by_pairs(f.G,t,ix),std::invalid_argument);
}
TEST(StreamAPI, OwnedSolverIgnoresSharedNdaSynchronizationSetting) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  auto reference=baseline(f);
  auto t=topology(3,"crossing");
  auto ix=select_indices(f.cpu->get_num_self_energy_backbones(t),2,1729);
  auto expected=reference.evaluate("se",t,ix);
  struct Guard {
    bool previous=nda::lapack::device::get_synchronization();
    ~Guard(){nda::lapack::device::set_synchronization(previous);}
  } guard;
  nda::lapack::device::set_synchronization(false);
  xca_experiments::serial_stream::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto value=candidate.compute_self_energy(f.G,t,ix);
  EXPECT_NO_THROW(compare(tensor(value[0].data()),expected));
  EXPECT_FALSE(nda::lapack::device::get_synchronization());
}

#include <gtest/gtest.h>
#include "../../../benchmarks/dense_cuda_small/harness.hpp"
#include "diagram_evaluator.hpp"
TEST(GraphAPI, InvalidPrefixRecoveryAndLongIndices) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  xca_experiments::graphs::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
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
TEST(GraphAPI, OwnedSolverIgnoresSharedNdaSynchronizationSetting) {
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
  xca_experiments::graphs::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto value=candidate.compute_self_energy(f.G,t,ix);
  EXPECT_NO_THROW(compare(tensor(value[0].data()),expected));
  EXPECT_FALSE(nda::lapack::device::get_synchronization());
}

TEST(GraphAPI, CacheKeysUseContentsAndEmptyDoesNotReplay) {
  using namespace dense_cuda_bench;
  Fixture f(Options{}); auto reference=baseline(f);
  xca_experiments::graphs::DiagramEvaluator candidate(f.options.beta,1e-12,f.itops,f.poles,f.coefficients,*f.fset);
  auto t=topology(6,"crossing");
  auto ix=select_indices(candidate.get_num_self_energy_backbones(t),3,1729);
  auto check=[&]{auto g=candidate.compute_self_energy(f.G,t,ix); compare(tensor(g[0].data()),reference.evaluate("se",t,ix));};
  check(); EXPECT_EQ(candidate.cache_builds(),1); check(); EXPECT_EQ(candidate.cache_builds(),1);
  auto pointer=ix.data(); ++ix(0); check(); EXPECT_EQ(pointer,ix.data()); EXPECT_EQ(candidate.cache_builds(),2);
  auto tp=t.data(); t()=topology(6,"chain"); check(); EXPECT_EQ(tp,t.data()); EXPECT_EQ(candidate.cache_builds(),3);
  nda::vector<long> empty(0); auto z=candidate.compute_self_energy(f.G,t,empty);
  EXPECT_EQ(nda::max_element(nda::abs(z[0].data())),0.); EXPECT_EQ(candidate.cache_builds(),3);
  check(); EXPECT_EQ(candidate.cache_builds(),3);
  nda::vector<long> too_many(129); EXPECT_THROW(candidate.compute_self_energy(f.G,t,too_many),std::invalid_argument);
  check(); EXPECT_EQ(candidate.cache_builds(),3);
}

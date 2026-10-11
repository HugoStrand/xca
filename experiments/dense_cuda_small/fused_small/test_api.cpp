#include "../../../benchmarks/dense_cuda_small/harness.hpp"
#include "diagram_evaluator.hpp"
#include <gtest/gtest.h>
TEST(BatchAPI, InvalidPrefixRecoveryAndLongIndices) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  xca_experiments::fused_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  auto reference = baseline(f);
  auto t         = topology(12, "crossing");
  long total     = candidate.get_num_self_energy_backbones(t);
  auto ix        = select_indices(total, 3, 1729);
  ASSERT_GT(total, long(INT_MAX) + 17);
  ix(0) = long(INT_MAX) + 17;
  nda::vector<long> bad{ix(0), total};
  EXPECT_THROW(candidate.compute_self_energy(f.G, t, bad), std::exception);
  auto value = candidate.compute_self_energy(f.G, t, ix);
  EXPECT_NO_THROW(compare(tensor(value[0].data()), reference.evaluate("se", t, ix)));
  nda::vector<long> empty(0);
  auto zero = candidate.compute_self_energy(f.G, t, empty);
  EXPECT_EQ(nda::max_element(nda::abs(zero[0].data())), 0.);
}
TEST(BatchAPI, OwnedSolverIgnoresSharedNdaSynchronizationSetting) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  auto reference = baseline(f);
  auto t         = topology(3, "crossing");
  auto ix        = select_indices(f.cpu->get_num_self_energy_backbones(t), 2, 1729);
  auto expected  = reference.evaluate("se", t, ix);
  struct Guard {
    bool previous = nda::lapack::device::get_synchronization();
    ~Guard() { nda::lapack::device::set_synchronization(previous); }
  } guard;
  nda::lapack::device::set_synchronization(false);
  xca_experiments::fused_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  auto value = candidate.compute_self_energy(f.G, t, ix);
  EXPECT_NO_THROW(compare(tensor(value[0].data()), expected));
  EXPECT_FALSE(nda::lapack::device::get_synchronization());
}

TEST(BatchAPI, TailsAndContentCacheWithDeterministicAccumulation) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  auto reference = baseline(f);
  for (int B : {1, 8, 32, 128}) {
    SCOPED_TRACE(B);
    xca_experiments::fused_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset, B);
    auto t     = topology(6, "crossing");
    long total = candidate.get_num_self_energy_backbones(t);
    for (int count : {1, B, B + 1, 2 * B + 3}) {
      auto ix           = select_indices(total, count, 1729);
      ix(ix.size() - 1) = ix(0);
      auto check        = [&] {
        auto got = candidate.compute_self_energy(f.G, t, ix);
        compare(tensor(got[0].data()), reference.evaluate("se", t, ix));
      };
      check();
      check();
      auto pointer = ix.data();
      ++ix(0);
      check();
      EXPECT_EQ(pointer, ix.data());
    }
    auto ix    = select_indices(total, B, 31415);
    auto check = [&] {
      auto got = candidate.compute_self_energy(f.G, t, ix);
      compare(tensor(got[0].data()), reference.evaluate("se", t, ix));
    };
    check();
    auto builds = candidate.cache_builds();
    check();
    EXPECT_EQ(candidate.cache_builds(), builds);
    auto gp = f.G[0].data().data();
    f.G[0].data() *= complex{.99, .01};
    check();
    EXPECT_EQ(gp, f.G[0].data().data());
    EXPECT_EQ(candidate.cache_builds(), builds);
    auto tp = t.data();
    t()     = topology(6, "chain");
    check();
    EXPECT_EQ(tp, t.data());
    EXPECT_GT(candidate.cache_builds(), builds);
    nda::vector<long> empty(0);
    auto zero = candidate.compute_self_energy(f.G, t, empty);
    EXPECT_EQ(nda::max_element(nda::abs(zero[0].data())), 0.);
    check();
    candidate.set_cache_enabled(false);
    check();
    builds = candidate.cache_builds();
    check();
    EXPECT_GT(candidate.cache_builds(), builds);
  }
}
TEST(FusedAPI, DenseComplexBasisAndChangedPropagator) {
  using namespace dense_cuda_bench;
  Fixture f(Options{});
  nda::matrix<complex> U(4, 4), Ud(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) U(i, j) = std::exp(complex{0., 2. * std::numbers::pi * i * j / 4. + .23 * j * j}) / 2.;
  Ud = nda::conj(nda::transpose(U));
  for (int a = 0; a < 2; ++a) {
    f.Fs(a, all, all) = nda::linalg::matmul(U, nda::linalg::matmul(nda::make_regular(f.Fs(a, all, all)), Ud));
    f.Fd(a, all, all) = nda::conj(nda::transpose(f.Fs(a, all, all)));
  }
  for (long t = 0; t < f.itops.rank(); ++t)
    f.G[0].data()(t, all, all) = nda::linalg::matmul(U, nda::linalg::matmul(nda::make_regular(f.G[0].data()(t, all, all)), Ud));
  f.fset         = std::make_unique<triqs_xca::dense::FSet>(f.Fs, f.Fd, f.coefficients);
  f.cpu          = std::make_unique<triqs_xca::dense::DiagramEvaluator>(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset);
  auto reference = baseline(f);
  for (int B : {1, 8, 128}) {
    xca_experiments::fused_small::DiagramEvaluator candidate(f.options.beta, 1e-12, f.itops, f.poles, f.coefficients, *f.fset, B);
    for (int order : {3, 6, 8, 12}) {
      SCOPED_TRACE(::testing::Message() << "B=" << B << " order=" << order);
      auto t   = topology(order, "crossing");
      auto ix  = select_indices(candidate.get_num_self_energy_backbones(t), 3, 31415);
      auto run = [&] {
        auto actual = candidate.compute_self_energy(f.G, t, ix);
        EXPECT_NO_THROW(compare(tensor(actual[0].data()), reference.evaluate("se", t, ix)));
      };
      run();
      auto pointer = f.G[0].data().data();
      f.G[0].data() *= complex{.99, .001};
      run();
      EXPECT_EQ(pointer, f.G[0].data().data());
    }
  }
}

#include <gtest/gtest.h>
#include "../../../benchmarks/dense_cuda_small/harness.hpp"
#include <limits>
TEST(DenseBenchmark, RejectsNonfiniteValuesAndMetrics) {
  using namespace dense_cuda_bench;
  tensor reference=nda::zeros<complex>(1,1,1), actual(reference);
  EXPECT_NO_THROW(compare(actual,reference));
  actual(0,0,0)={std::numeric_limits<double>::quiet_NaN(),0.};
  EXPECT_THROW(compare(actual,reference,false),std::runtime_error);
  actual(0,0,0)={std::numeric_limits<double>::infinity(),0.};
  EXPECT_THROW(compare(actual,reference,false),std::runtime_error);
  reference(0,0,0)={1.7e308,1.7e308}; actual=reference;
  EXPECT_THROW(compare(actual,reference,false),std::runtime_error);
  reference()=complex{0.,0.}; actual()=complex{1e-250,0.};
  EXPECT_THROW(compare(actual,reference),std::runtime_error);
}

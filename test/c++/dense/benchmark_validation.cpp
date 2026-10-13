#include <gtest/gtest.h>
#include "../../../benchmarks/dense_common/harness.hpp"
#include <limits>
#include <triqs_xca/dense/cuda_small/qualification.hpp>
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

TEST(DenseBenchmark, UnknownEnvironmentsAreNotQualified) {
  using namespace triqs_xca::dense::cuda_small;
  QualificationEnvironment local{"NVIDIA RTX A4000",8,6,13040,13040,13000,{13,8,0},{12,3,4}};
  ASSERT_TRUE(qualified_environment(local));
  auto changed=local;
  changed.gpu_name="NVIDIA GH200"; changed.major=9; changed.minor=0;
  EXPECT_FALSE(qualified_environment(changed)); // Target hardware remains unqualified.
  changed=local; ++changed.cublas[1]; EXPECT_FALSE(qualified_environment(changed));
  changed=local; ++changed.cusolver[2]; EXPECT_FALSE(qualified_environment(changed));
  changed=local; ++changed.runtime; EXPECT_FALSE(qualified_environment(changed));
  changed=local; ++changed.compiled_runtime; EXPECT_FALSE(qualified_environment(changed));
  changed=local; ++changed.driver; EXPECT_FALSE(qualified_environment(changed));
}

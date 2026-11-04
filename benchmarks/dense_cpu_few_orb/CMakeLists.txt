add_executable(dense_cpu_few_orb EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(dense_cpu_few_orb PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

# Eigen is an optional benchmark dependency, never a requirement of the evaluator.
find_path(XCA_BENCHMARK_EIGEN_INCLUDE_DIR Eigen/Core PATH_SUFFIXES eigen3)
if(XCA_BENCHMARK_EIGEN_INCLUDE_DIR)
  add_executable(dense_few_orb_eigen_gemm EXCLUDE_FROM_ALL eigen_gemm.cpp)
  target_include_directories(dense_few_orb_eigen_gemm SYSTEM PRIVATE ${XCA_BENCHMARK_EIGEN_INCLUDE_DIR})
  target_link_libraries(dense_few_orb_eigen_gemm PRIVATE triqs_xca_c ${PROJECT_NAME}_warnings)
endif()

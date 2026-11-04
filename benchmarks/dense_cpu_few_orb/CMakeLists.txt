add_executable(dense_cpu_few_orb EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(dense_cpu_few_orb PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

# Uses the same Eigen dependency and kernels as the production evaluator.
add_executable(dense_few_orb_eigen_gemm EXCLUDE_FROM_ALL eigen_gemm.cpp)
target_link_libraries(dense_few_orb_eigen_gemm PRIVATE triqs_xca_c ${PROJECT_NAME}_warnings)

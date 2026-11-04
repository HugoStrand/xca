# CPU batching experiment, using the same physical model helpers as evaluator tests.
add_executable(dense_cpu_batched EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(dense_cpu_batched PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

add_subdirectory(experiments/small_n)

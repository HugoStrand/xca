set(sets_dir "${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_sets")
set(scalar_dir "${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_paths")
add_executable(block_sparse_cpu_density_signs_benchmark "${sets_dir}/density_benchmark.cpp")
target_compile_definitions(block_sparse_cpu_density_signs_benchmark PRIVATE XCA_DENSITY_SETS_CPU_BATCHED
  XCA_EXPERIMENT_SCALAR_SETS XCA_EXPERIMENT_SCALAR_CONSTANTS XCA_EXPERIMENT_SCALAR_SIGN_HOIST)
target_link_libraries(block_sparse_cpu_density_signs_benchmark PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c ${PROJECT_NAME}_warnings)
if(NOT TARGET block_sparse_scalar_sets_evaluator)
  return()
endif()
set(generated_dir "${PROJECT_BINARY_DIR}/benchmarks/experiments/block_sparse_cuda/scalar_paths")
add_library(block_sparse_scalar_signs_evaluator STATIC "${generated_dir}/scalar_evaluator.cpp")
target_include_directories(block_sparse_scalar_signs_evaluator PUBLIC "${generated_dir}"
  "${PROJECT_SOURCE_DIR}/c++/triqs_xca/block_sparse/cuda" PRIVATE "${scalar_dir}")
target_compile_definitions(block_sparse_scalar_signs_evaluator PRIVATE XCA_EXPERIMENT_SCALAR_FANIN
  XCA_EXPERIMENT_SCALAR_ARENA XCA_EXPERIMENT_SCALAR_CONSTANTS XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY
  XCA_EXPERIMENT_SCALAR_SETS XCA_EXPERIMENT_SCALAR_SIGN_HOIST)
target_link_libraries(block_sparse_scalar_signs_evaluator PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c block_sparse_scalar_edge_kernels)
add_executable(block_sparse_scalar_density_signs_benchmark "${sets_dir}/density_benchmark.cpp")
target_compile_definitions(block_sparse_scalar_density_signs_benchmark PRIVATE XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY XCA_EXPERIMENT_SCALAR_SIGN_HOIST)
target_link_libraries(block_sparse_scalar_density_signs_benchmark PRIVATE block_sparse_scalar_signs_evaluator ${PROJECT_NAME}_warnings)
if(Build_Tests)
  add_executable(block_sparse_scalar_signs_tests "${generated_dir}/scalar_path_tests.cpp"
    "${scalar_dir}/qualification.cpp" "${scalar_dir}/fanin_order.cpp" "${scalar_dir}/constants_layout.cpp" "${sets_dir}/qualification.cpp")
  target_compile_definitions(block_sparse_scalar_signs_tests PRIVATE XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY XCA_EXPERIMENT_SCALAR_SETS XCA_EXPERIMENT_SCALAR_SIGN_HOIST)
  target_link_libraries(block_sparse_scalar_signs_tests PRIVATE block_sparse_scalar_signs_evaluator ${PROJECT_NAME}_test_utils gtest_main)
  add_test(NAME "experiments/block_sparse_cuda/scalar_signs" COMMAND block_sparse_scalar_signs_tests)
endif()

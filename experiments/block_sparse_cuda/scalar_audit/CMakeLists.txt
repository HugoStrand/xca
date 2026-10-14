if(NOT TARGET block_sparse_scalar_staging_evaluator)
  return()
endif()
set(sets_dir "${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_sets")
set(scalar_dir "${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_paths")
set(generated_dir "${PROJECT_BINARY_DIR}/benchmarks/experiments/block_sparse_cuda/scalar_paths")
add_library(block_sparse_scalar_audit_evaluator STATIC "${generated_dir}/scalar_evaluator.cpp")
target_include_directories(block_sparse_scalar_audit_evaluator PUBLIC "${generated_dir}"
  "${PROJECT_SOURCE_DIR}/c++/triqs_xca/block_sparse/cuda" PRIVATE "${scalar_dir}" "${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_resident" "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(block_sparse_scalar_audit_evaluator PRIVATE XCA_EXPERIMENT_SCALAR_FANIN
  XCA_EXPERIMENT_SCALAR_ARENA XCA_EXPERIMENT_SCALAR_CONSTANTS XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY
  XCA_EXPERIMENT_SCALAR_SETS XCA_EXPERIMENT_SCALAR_SIGN_HOIST XCA_EXPERIMENT_SCALAR_RESIDENT XCA_EXPERIMENT_SCALAR_STAGING XCA_EXPERIMENT_SCALAR_STREAMS XCA_EXPERIMENT_SCALAR_AUDIT)
target_link_libraries(block_sparse_scalar_audit_evaluator PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c block_sparse_scalar_edge_kernels block_sparse_resident_weight_kernels)
add_executable(block_sparse_scalar_cutoff_audit audit.cpp)
target_compile_definitions(block_sparse_scalar_cutoff_audit PRIVATE XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY)
target_link_libraries(block_sparse_scalar_cutoff_audit PRIVATE block_sparse_scalar_audit_evaluator ${PROJECT_NAME}_warnings)
add_executable(block_sparse_scalar_stage_audit stages.cpp)
target_link_libraries(block_sparse_scalar_stage_audit PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c block_sparse_scalar_edge_kernels ${PROJECT_NAME}_warnings)

# Performance workloads are independent of examples and the test suite.
function(xca_add_benchmark name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c ${PROJECT_NAME}_warnings)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

add_subdirectory(dense_cpu)
add_subdirectory(dense_cuda)
add_subdirectory(dense_cuda_small)

add_subdirectory(block_sparse_cpu)
add_subdirectory(block_sparse_cuda)

# Keep optimization trials separate from retained evaluator baselines.
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/dlr_hoisted
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/dlr_hoisted)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/operator_apply
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/operator_apply)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_edges
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_edges)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_paths
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_paths)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_cutoff
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_cutoff)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_sets
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_sets)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_signs
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_signs)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_resident
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_resident)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_staging
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_staging)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_streams
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_streams)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_real_nca
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_real_nca)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/scalar_audit
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/scalar_audit)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/special_vertex
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/special_vertex)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/mapped_special
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/mapped_special)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/compact_dlr
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/compact_dlr)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/nca_admission
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/nca_admission)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/topology_reuse
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/topology_reuse)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/output_sectors
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/output_sectors)

add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/convolution_rows
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/convolution_rows)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/phase_liveness
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/phase_liveness)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/host_inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/host_inputs)
add_subdirectory(${PROJECT_SOURCE_DIR}/experiments/block_sparse_cuda/stream_fragments
                 ${CMAKE_CURRENT_BINARY_DIR}/experiments/block_sparse_cuda/stream_fragments)

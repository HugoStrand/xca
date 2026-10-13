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

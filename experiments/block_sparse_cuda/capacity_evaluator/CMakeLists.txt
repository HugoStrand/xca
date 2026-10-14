if(NOT nda_WITH_CUDA_SUPPORT)
  return()
endif()
add_executable(block_sparse_capacity_evaluator probe.cpp)
target_link_libraries(block_sparse_capacity_evaluator PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c)
if(Build_Tests)
  add_test(NAME "experiments/block_sparse_cuda/capacity_evaluator"
           COMMAND block_sparse_capacity_evaluator --modes 4 --fields 4)
endif()

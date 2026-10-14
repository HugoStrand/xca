add_executable(block_sparse_topology_reuse_probe probe.cpp)
target_link_libraries(block_sparse_topology_reuse_probe PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c ${PROJECT_NAME}_warnings)

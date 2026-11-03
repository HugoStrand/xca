# Isolated numerical replay experiments; deliberately outside the default build.
add_executable(block_sparse_layout EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(block_sparse_layout PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

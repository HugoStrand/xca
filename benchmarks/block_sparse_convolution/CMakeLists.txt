# Whole prepared-backbone calls, including path discovery and result ownership.
add_executable(block_sparse_convolution EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(block_sparse_convolution PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

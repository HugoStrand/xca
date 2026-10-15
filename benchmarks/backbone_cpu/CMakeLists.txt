# Uses the same physical fixtures as the C++ evaluator tests. Built only on request.
add_executable(backbone_cpu EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(backbone_cpu PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

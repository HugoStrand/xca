add_executable(dense_cpu_few_orb EXCLUDE_FROM_ALL main.cpp)
target_link_libraries(dense_cpu_few_orb PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

# Uses the same Eigen dependency and kernels as the production evaluator.
add_executable(dense_few_orb_eigen_gemm EXCLUDE_FROM_ALL eigen_gemm.cpp)
target_link_libraries(dense_few_orb_eigen_gemm PRIVATE triqs_xca_c ${PROJECT_NAME}_warnings)

# Controlled cache-policy comparison; numerical evaluation stays in sync with
# the production source. The generated alternatives live only in the build tree.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(few_orb_source ${PROJECT_SOURCE_DIR}/c++/triqs_xca/dense)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ordered_linear.hpp ${CMAKE_CURRENT_BINARY_DIR}/ordered_linear.cpp
         ${CMAKE_CURRENT_BINARY_DIR}/canonical_linear.hpp ${CMAKE_CURRENT_BINARY_DIR}/canonical_linear.cpp
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_cache_variants.py
          ${few_orb_source} ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS make_cache_variants.py ${few_orb_source}/diagram_evaluator_few_orb.hpp ${few_orb_source}/diagram_evaluator_few_orb.cpp
  VERBATIM)
add_executable(dense_few_orb_cache_lookup EXCLUDE_FROM_ALL cache_lookup.cpp
               ${CMAKE_CURRENT_BINARY_DIR}/ordered_linear.cpp ${CMAKE_CURRENT_BINARY_DIR}/canonical_linear.cpp)
target_include_directories(dense_few_orb_cache_lookup PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${few_orb_source})
target_link_libraries(dense_few_orb_cache_lookup PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

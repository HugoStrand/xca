# Isolated research targets. Production evaluators and convolution are unchanged.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(evaluator_source ${PROJECT_SOURCE_DIR}/c++/triqs_xca/dense)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/probe_evaluator.hpp ${CMAKE_CURRENT_BINARY_DIR}/probe_evaluator.cpp
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_probe.py
          ${evaluator_source} ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS make_probe.py ${evaluator_source}/diagram_evaluator_batched.hpp ${evaluator_source}/diagram_evaluator_batched.cpp
  VERBATIM)

add_executable(dense_small_n_evaluator EXCLUDE_FROM_ALL evaluator_probe.cpp ${CMAKE_CURRENT_BINARY_DIR}/probe_evaluator.cpp)
target_include_directories(dense_small_n_evaluator PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dense_small_n_evaluator PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)

foreach(probe convolution convmat recipe_reuse)
  if(probe STREQUAL recipe_reuse)
    set(probe_source recipe_reuse.cpp)
  else()
    set(probe_source ${probe}_probe.cpp)
  endif()
  add_executable(dense_small_n_${probe} EXCLUDE_FROM_ALL ${probe_source})
  target_link_libraries(dense_small_n_${probe} PRIVATE triqs_xca_test_utils gtest ${PROJECT_NAME}_warnings)
endforeach()

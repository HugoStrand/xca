cmake_minimum_required(VERSION 3.22)
project(xca_graphs LANGUAGES CXX)
find_package(TRIQS 4.0 REQUIRED)
find_package(cppdlr REQUIRED)
get_filename_component(XCA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(XCA_BASELINE_BUILD "${XCA_ROOT}/cbuild-cuda" CACHE PATH "Frozen CUDA build")
add_library(graphs STATIC dlr_ops.cpp diagram_evaluator.cpp ../serial_stream/device_ops.cpp)
target_include_directories(graphs PUBLIC "${XCA_ROOT}/c++")
target_compile_features(graphs PUBLIC cxx_std_23)
target_link_libraries(graphs PUBLIC triqs cppdlr::cppdlr_c)
add_executable(graphs_benchmark benchmark.cpp)
target_link_libraries(graphs_benchmark PRIVATE graphs "${XCA_BASELINE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a"
 "${XCA_BASELINE_BUILD}/c++/triqs_soehyb/libtriqs_soehyb_c.a")
# Reuse existing analytic/rotation tests; the adapter replaces only ordinary SE.
enable_testing()
set(TEST_ROOT "${XCA_ROOT}/test/c++")
foreach(name one_fermion_se_spgf_all_evals two_fermions_se_spgf_all_evals unitary_invariance_all_evals)
 file(READ "${TEST_ROOT}/${name}.cpp" contents)
 string(REPLACE "dense::cuda::DiagramEvaluator" "GraphTestAdapter" contents "${contents}")
 file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp" "#include \"test_adapter.hpp\"\n${contents}")
 add_executable(graph_${name} "${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp")
 target_include_directories(graph_${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TEST_ROOT}"
   "${XCA_BASELINE_BUILD}/deps/GTest_src/googletest/include")
 target_link_libraries(graph_${name} PRIVATE graphs
   "${XCA_BASELINE_BUILD}/lib/libgtest_main.a"
   "${XCA_BASELINE_BUILD}/test/c++/libtriqs_xca_test_utils.a"
   "${XCA_BASELINE_BUILD}/lib/libgtest.a"
   "${XCA_BASELINE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a")
 add_test(NAME graph_${name} COMMAND graph_${name})
endforeach()
add_executable(graph_api test_api.cpp)
target_include_directories(graph_api PRIVATE "${XCA_BASELINE_BUILD}/deps/GTest_src/googletest/include")
target_link_libraries(graph_api PRIVATE graphs "${XCA_BASELINE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a"
 "${XCA_BASELINE_BUILD}/lib/libgtest_main.a" "${XCA_BASELINE_BUILD}/lib/libgtest.a")
add_test(NAME graph_api COMMAND graph_api)

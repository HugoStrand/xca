cmake_minimum_required(VERSION 3.22)
project(xca_serial_stream LANGUAGES CXX)
find_package(TRIQS 4.0 REQUIRED)
find_package(cppdlr REQUIRED)
get_filename_component(XCA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(XCA_BASELINE_BUILD "${XCA_ROOT}/cbuild-cuda" CACHE PATH "Existing frozen CUDA build")
add_library(serial_stream STATIC device_ops.cpp dlr_ops.cpp diagram_evaluator.cpp)
target_include_directories(serial_stream PUBLIC "${XCA_ROOT}/c++")
target_compile_features(serial_stream PUBLIC cxx_std_23)
target_link_libraries(serial_stream PUBLIC triqs cppdlr::cppdlr_c)
add_executable(serial_stream_benchmark benchmark.cpp)
target_link_libraries(serial_stream_benchmark PRIVATE serial_stream
 "${XCA_BASELINE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a"
 "${XCA_BASELINE_BUILD}/c++/triqs_soehyb/libtriqs_soehyb_c.a")
# Reuse existing analytic/rotation tests; the adapter replaces only ordinary SE.
enable_testing()
set(TEST_ROOT "${XCA_ROOT}/test/c++")
foreach(name one_fermion_se_spgf_all_evals two_fermions_se_spgf_all_evals unitary_invariance_all_evals)
 file(READ "${TEST_ROOT}/${name}.cpp" contents)
 string(REPLACE "dense::cuda::DiagramEvaluator" "StreamTestAdapter" contents "${contents}")
 file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp" "#include \"test_adapter.hpp\"\n${contents}")
 add_executable(stream_${name} "${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp")
 target_include_directories(stream_${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TEST_ROOT}"
   "${XCA_BASELINE_BUILD}/deps/GTest_src/googletest/include")
 target_link_libraries(stream_${name} PRIVATE serial_stream
   "${XCA_BASELINE_BUILD}/lib/libgtest_main.a"
   "${XCA_BASELINE_BUILD}/test/c++/libtriqs_xca_test_utils.a"
   "${XCA_BASELINE_BUILD}/lib/libgtest.a"
   "${XCA_BASELINE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a")
 add_test(NAME stream_${name} COMMAND stream_${name})
endforeach()
add_executable(stream_dlr test_dlr.cpp)
target_include_directories(stream_dlr PRIVATE "${XCA_BASELINE_BUILD}/deps/GTest_src/googletest/include")
target_link_libraries(stream_dlr PRIVATE serial_stream "${XCA_BASELINE_BUILD}/lib/libgtest_main.a" "${XCA_BASELINE_BUILD}/lib/libgtest.a")
add_test(NAME stream_dlr COMMAND stream_dlr)
add_executable(stream_api test_api.cpp)
target_include_directories(stream_api PRIVATE "${XCA_BASELINE_BUILD}/deps/GTest_src/googletest/include")
target_link_libraries(stream_api PRIVATE serial_stream "${XCA_BASELINE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a"
 "${XCA_BASELINE_BUILD}/lib/libgtest_main.a" "${XCA_BASELINE_BUILD}/lib/libgtest.a")
add_test(NAME stream_api COMMAND stream_api)

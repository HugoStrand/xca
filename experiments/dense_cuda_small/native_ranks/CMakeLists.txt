cmake_minimum_required(VERSION 3.22)
project(xca_native_rank_probe LANGUAGES CXX)
find_package(TRIQS 4.0 REQUIRED)
find_package(cppdlr REQUIRED)
find_package(CUDAToolkit REQUIRED)
get_filename_component(XCA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(XCA_NATIVE_BUILD "${XCA_ROOT}/cbuild-cuda" CACHE PATH "Built native CUDA-small tree")
option(XCA_PROBE_PRODUCTS "Use tiled products for all N4/N8 ranks" ON)
set(native_source "${XCA_ROOT}/c++/triqs_xca/dense/cuda_small")
set(snapshot "${CMAKE_CURRENT_BINARY_DIR}/snapshot")
file(MAKE_DIRECTORY "${snapshot}")
# Generate a distinct host namespace. Reuse identical compiled kernels, preserving
# the arithmetic, stream, solver, metadata and resource policies of production.
file(GLOB inputs "${native_source}/*.hpp" "${native_source}/*.cpp" "${native_source}/*.cu")
foreach(input IN LISTS inputs)
  get_filename_component(name "${input}" NAME)
  file(READ "${input}" contents)
  string(REPLACE "triqs_xca::dense::cuda_small" "triqs_xca::dense::cuda_small_probe" contents "${contents}")
  if(name STREQUAL "diagram_evaluator.cpp")
    string(FIND "${contents}" "if (!qualified_shape(r_, N_)) return false;" guard)
    if(guard LESS 0)
      message(FATAL_ERROR "Native batching guard changed; review probe transformation")
    endif()
    string(REPLACE "if (!qualified_shape(r_, N_)) return false;"
      "if (N_ != 4 && N_ != 8) return false;" contents "${contents}")
  elseif(name STREQUAL "device_ops.cpp")
    string(FIND "${contents}" "mode_ = qualified_shape(rank, N_) ? 2 : 0;" guard)
    if(guard LESS 0)
      message(FATAL_ERROR "Native product guard changed; review probe transformation")
    endif()
    if(XCA_PROBE_PRODUCTS)
      set(probe_mode "mode_ = (N_ == 4 || N_ == 8) ? 2 : 0;")
    else()
      set(probe_mode "mode_ = 0;")
    endif()
    string(REPLACE "mode_ = qualified_shape(rank, N_) ? 2 : 0;"
      "${probe_mode}" contents "${contents}")
  endif()
  # This explicit experiment evaluates unqualified environments/shapes. Its
  # batch-counter assertion and strict reference gate must pass before promotion.
  string(REPLACE "if (!qualified_environment(environment)) mode_ = 0;"
    "// Environment admission intentionally bypassed by the explicit rank probe." contents "${contents}")
  string(REPLACE "if (dlr_.device_ops().mode() != 2) return false;"
    "// Product/library controls both exercise batching in this explicit probe." contents "${contents}")
  file(WRITE "${snapshot}/${name}" "${contents}")
endforeach()
if(XCA_PROBE_PRODUCTS)
  file(WRITE "${snapshot}/probe_config.hpp" "#define XCA_PROBE_PRODUCTS 1\n")
else()
  file(WRITE "${snapshot}/probe_config.hpp" "#define XCA_PROBE_PRODUCTS 0\n")
endif()
configure_file(benchmark.cpp "${snapshot}/benchmark.cpp" COPYONLY)
configure_file(CMakeLists.txt "${snapshot}/CMakeLists.txt" COPYONLY)
add_library(native_rank_probe STATIC "${snapshot}/device_ops.cpp" "${snapshot}/dlr_ops.cpp"
  "${snapshot}/batch_evaluator.cpp" "${snapshot}/diagram_evaluator.cpp")
target_include_directories(native_rank_probe PUBLIC "${snapshot}" "${XCA_ROOT}/c++" "${XCA_ROOT}")
target_compile_features(native_rank_probe PUBLIC cxx_std_23)
target_compile_definitions(native_rank_probe PUBLIC XCA_HAVE_CUDA_SMALL)
target_link_libraries(native_rank_probe PUBLIC triqs cppdlr::cppdlr_c CUDA::cudart
  "${XCA_NATIVE_BUILD}/c++/triqs_xca/dense/cuda_small/libtriqs_xca_small_kernels.a")
add_executable(native_rank_benchmark "${snapshot}/benchmark.cpp")
target_link_libraries(native_rank_benchmark PRIVATE native_rank_probe
  "${XCA_NATIVE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a")

# Reuse complete independent physics/API suites with the actual probe class.
# No adapter delegates observables or dynamic interactions to retained CUDA.
enable_testing()
foreach(source one_fermion_se_spgf_all_evals two_fermions_se_spgf_all_evals
    unitary_invariance_all_evals dynint_spin_rotation flat_index_range propagator_layout
    dense/dynint_parity dense/correlator_statistics)
  file(READ "${XCA_ROOT}/test/c++/${source}.cpp" contents)
  string(REPLACE "triqs_xca/dense/cuda/diagram_evaluator.hpp" "diagram_evaluator.hpp" contents "${contents}")
  string(REPLACE "dense::cuda::" "dense::cuda_small_probe::" contents "${contents}")
  string(REPLACE "NDA_HAVE_CUDA" "XCA_HAVE_CUDA_SMALL" contents "${contents}")
  string(REPLACE "/" "_" suffix "${source}")
  set(target "native_rank_${suffix}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp" "${contents}")
  add_executable(${target} "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp")
  target_include_directories(${target} PRIVATE "${XCA_ROOT}/test/c++" "${XCA_NATIVE_BUILD}/deps/GTest_src/googletest/include")
  target_link_libraries(${target} PRIVATE native_rank_probe
    "${XCA_NATIVE_BUILD}/test/c++/libtriqs_xca_test_utils.a"
    "${XCA_NATIVE_BUILD}/lib/libgtest_main.a" "${XCA_NATIVE_BUILD}/lib/libgtest.a"
    "${XCA_NATIVE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a")
  add_test(NAME ${target} COMMAND ${target})
endforeach()
add_executable(native_rank_channels test_channels.cpp)
target_include_directories(native_rank_channels PRIVATE "${XCA_ROOT}/test/c++" "${XCA_NATIVE_BUILD}/deps/GTest_src/googletest/include")
target_link_libraries(native_rank_channels PRIVATE native_rank_probe
  "${XCA_NATIVE_BUILD}/test/c++/libtriqs_xca_test_utils.a"
  "${XCA_NATIVE_BUILD}/lib/libgtest_main.a" "${XCA_NATIVE_BUILD}/lib/libgtest.a"
  "${XCA_NATIVE_BUILD}/c++/triqs_xca/libtriqs_xca_c.a")
add_test(NAME native_rank_channels COMMAND native_rank_channels)

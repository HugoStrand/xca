# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# Helper library of the tests, in namespace triqs_xca::test_utils
file(GLOB test_utils_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test_utils/*.cpp)
add_library(${PROJECT_NAME}_test_utils STATIC ${test_utils_sources})
target_include_directories(${PROJECT_NAME}_test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}_test_utils PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c PRIVATE ${PROJECT_NAME}_warnings)

# List of all tests
file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
list(FILTER all_tests EXCLUDE REGEX "^test_utils/")

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  # Tests in a subdirectory are listed as <dir>/<name> in ctest, with the target <dir>_<name>
  if(test_dir)
    set(test_label ${test_dir}/${test_name})
    string(REPLACE "/" "_" test_target ${test_dir}_${test_name})
  else()
    set(test_label ${test_name})
    set(test_target ${test_name})
  endif()
  add_executable(${test_target} ${test})
  set_target_properties(${test_target} PROPERTIES OUTPUT_NAME ${test_name})
  target_link_libraries(${test_target} ${PROJECT_NAME}::${PROJECT_NAME}_c ${PROJECT_NAME}_warnings gtest_main)
  if(test_dir STREQUAL "triqs_soehyb")
    target_link_libraries(${test_target} ${PROJECT_NAME}::triqs_soehyb_c)
  else()
    target_link_libraries(${test_target} ${PROJECT_NAME}_test_utils)
  endif()
  set_property(TARGET ${test_target} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  add_test(NAME ${test_label} COMMAND ${test_target} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    set_target_properties(${test_target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  endif()
  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    add_custom_command(
      TARGET ${test_target}
      COMMAND ${CPPCHECK_EXECUTABLE}
      --enable=warning,style,performance,portability
      --std=c++20
      --template=gcc
      --verbose
      --force
      --quiet
      ${CMAKE_CURRENT_SOURCE_DIR}/${test}
    )
  endif()
endforeach()

# Exercise the complete native small evaluator with the same independent physics
# and API checks as library CUDA. Each executable owns its GoogleTest registry.
if(XCA_WITH_CUDA_SMALL)
  foreach(source one_fermion_se_spgf_all_evals two_fermions_se_spgf_all_evals
      unitary_invariance_all_evals dynint_spin_rotation flat_index_range propagator_layout
      dense/dynint_parity dense/correlator_statistics)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}.cpp" contents)
    string(REPLACE "dense/cuda/" "dense/cuda_small/" contents "${contents}")
    string(REPLACE "dense::cuda::" "dense::cuda_small::" contents "${contents}")
    string(REPLACE "NDA_HAVE_CUDA" "XCA_HAVE_CUDA_SMALL" contents "${contents}")
    string(REPLACE "/" "_" suffix "${source}")
    set(target "cuda_small_${suffix}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp" "${contents}")
    add_executable(${target} "${CMAKE_CURRENT_BINARY_DIR}/${target}.cpp")
    target_link_libraries(${target} PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c
      ${PROJECT_NAME}_test_utils ${PROJECT_NAME}_warnings gtest_main)
    add_test(NAME "dense/cuda_small/${suffix}" COMMAND ${target})
  endforeach()
endif()

# Keep the library control and run identical physics/API gates with ordered CSR.
if(TARGET triqs_xca_sparse_kernels)
  set(reference "${CMAKE_CURRENT_SOURCE_DIR}/block_sparse/cuda_diagram_evaluator.cpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${reference}")
  file(READ "${reference}" contents)
  string(REPLACE "gpu::DiagramEvaluator" "OrderedTestEvaluator" contents "${contents}")
  set(adapter "class OrderedTestEvaluator : public gpu::DiagramEvaluator {
    public:
    OrderedTestEvaluator(gpu::OperatorModel model, nda::vector_const_view<double> poles,
      triqs::mesh::dlr_imtime mesh, std::size_t budget = 1024ULL*1024*1024,
      std::size_t sparse_budget = 32ULL*1024*1024)
      : gpu::DiagramEvaluator(std::move(model), poles, std::move(mesh), budget,
                             sparse_budget, gpu::SparseBackend::Ordered) {}
  };")
  string(REPLACE "namespace gpu = bs::cuda;" "namespace gpu = bs::cuda;\n${adapter}" contents "${contents}")
  file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/block_sparse_ordered_evaluator.cpp" CONTENT "${contents}" @ONLY)
  add_executable(block_sparse_native_evaluator "${CMAKE_CURRENT_BINARY_DIR}/block_sparse_ordered_evaluator.cpp")
  target_link_libraries(block_sparse_native_evaluator PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c ${PROJECT_NAME}_test_utils gtest_main)
  add_test(NAME "block_sparse/ordered_evaluator" COMMAND block_sparse_native_evaluator)
endif()

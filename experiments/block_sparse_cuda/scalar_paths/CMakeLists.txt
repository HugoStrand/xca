if(NOT TARGET block_sparse_scalar_edge_kernels)
  return()
endif()
set(source_dir "${PROJECT_SOURCE_DIR}/c++/triqs_xca/block_sparse/cuda")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_dir}/diagram_evaluator.hpp" "${source_dir}/diagram_evaluator.cpp")
file(READ "${source_dir}/diagram_evaluator.hpp" header)
file(READ "${source_dir}/diagram_evaluator.cpp" source)
string(REPLACE "DiagramEvaluator" "ScalarPathEvaluator" header "${header}")
string(REPLACE "DiagramEvaluator" "ScalarPathEvaluator" source "${source}")
string(REPLACE "\"diagram_evaluator.hpp\"" "\"scalar_evaluator.hpp\"\n#include \"../scalar_edges/scalar_pack.hpp\"" source "${source}")
string(REPLACE "    void evaluate(std::vector<Tensor>" "#include \"scheduler.inc\"\n    void evaluate(std::vector<Tensor>" source "${source}")
string(REPLACE "if (first != last) x.admit();" "if (first != last) x.admit(checked_add(x.bound(),x.scalar_extra_bound()));" source "${source}")
set(original "      x.counts = for_each_self_energy_path(x.model, x.hyb->poles, topology, live, first, last,
                                           [&](Backbone &backbone, SelfEnergyPath const &path) { x.evaluate(input, output, backbone, path); });")
string(FIND "${source}" "${original}" position)
if(position EQUAL -1)
  message(FATAL_ERROR "Scalar scheduler cannot identify original path visitor")
endif()
set(replacement "      if (x.scalar_eligible()) {
        std::unique_ptr<scalar_edge_experiment::ScalarPack> packed;
        std::vector<Impl::ScalarWork> work;work.reserve(512);
        auto flush=[&]{if(work.empty())return;if(!packed)packed=std::make_unique<scalar_edge_experiment::ScalarPack>(x.ops,x.mesh.dlr_it(),512);x.scalar_flush(input,output,work,*packed);};
        x.counts=for_each_self_energy_path(x.model,x.hyb->poles,topology,live,first,last,[&](Backbone &backbone,SelfEnergyPath const &path){
          if(!work.empty() && work.front().second.special_vertex!=path.special_vertex)flush();
          work.emplace_back(backbone,path);if(work.size()==512)flush();
        });
        flush();
      } else {
${original}
      }")
string(REPLACE "${original}" "${replacement}" source "${source}")
string(REPLACE "self_energy_storage_bound() const { return impl_->bound(); }"
  "self_energy_storage_bound() const { return checked_add(impl_->bound(),impl_->scalar_extra_bound()); }" source "${source}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/scalar_evaluator.hpp" CONTENT "${header}" @ONLY)
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/scalar_evaluator.cpp" CONTENT "${source}" @ONLY)
add_library(block_sparse_scalar_path_evaluator STATIC "${CMAKE_CURRENT_BINARY_DIR}/scalar_evaluator.cpp")
target_include_directories(block_sparse_scalar_path_evaluator PUBLIC "${CMAKE_CURRENT_BINARY_DIR}" "${source_dir}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(block_sparse_scalar_path_evaluator PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c block_sparse_scalar_edge_kernels)
xca_add_benchmark(block_sparse_scalar_path_benchmark)
target_link_libraries(block_sparse_scalar_path_benchmark PRIVATE block_sparse_scalar_path_evaluator)
if(Build_Tests)
  set(reference "${PROJECT_SOURCE_DIR}/test/c++/block_sparse/cuda_diagram_evaluator.cpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${reference}")
  file(READ "${reference}" tests)
  string(REPLACE "<triqs_xca/block_sparse/cuda/diagram_evaluator.hpp>" "\"scalar_evaluator.hpp\"" tests "${tests}")
  string(REPLACE "gpu::DiagramEvaluator" "gpu::ScalarPathEvaluator" tests "${tests}")
  file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/scalar_path_tests.cpp" CONTENT "${tests}" @ONLY)
  add_executable(block_sparse_scalar_path_tests "${CMAKE_CURRENT_BINARY_DIR}/scalar_path_tests.cpp")
  target_link_libraries(block_sparse_scalar_path_tests PRIVATE block_sparse_scalar_path_evaluator ${PROJECT_NAME}_test_utils gtest_main)
  add_test(NAME "experiments/block_sparse_cuda/scalar_paths" COMMAND block_sparse_scalar_path_tests)
endif()

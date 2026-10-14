xca_add_benchmark(block_sparse_cpu_scalar_path_benchmark)
add_executable(block_sparse_cpu_scalar_constants_benchmark block_sparse_cpu_scalar_path_benchmark.cpp)
target_compile_definitions(block_sparse_cpu_scalar_constants_benchmark PRIVATE XCA_EXPERIMENT_SCALAR_CONSTANTS)
target_link_libraries(block_sparse_cpu_scalar_constants_benchmark PRIVATE ${PROJECT_NAME}::${PROJECT_NAME}_c ${PROJECT_NAME}_warnings)
if(NOT TARGET block_sparse_scalar_edge_kernels)
  return()
endif()
set(source_dir "${PROJECT_SOURCE_DIR}/c++/triqs_xca/block_sparse/cuda")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_dir}/diagram_evaluator.hpp" "${source_dir}/diagram_evaluator.cpp")
file(READ "${source_dir}/diagram_evaluator.hpp" header)
file(READ "${source_dir}/diagram_evaluator.cpp" source)
string(REPLACE "      if (placement != OutputPlacement::Resident && placement != OutputPlacement::HostSectors)"
  "      if (placement != OutputPlacement::Resident) throw std::invalid_argument(\"Scalar experiment requires resident output\");\n      if (placement != OutputPlacement::Resident && placement != OutputPlacement::HostSectors)" source "${source}")
string(REPLACE "DiagramEvaluator" "ScalarPathEvaluator" header "${header}")
string(REPLACE "DiagramEvaluator" "ScalarPathEvaluator" source "${source}")
string(REPLACE "std::size_t last_contributing_paths() const;"
  "std::size_t last_contributing_paths() const;\n#ifdef XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY\n    std::size_t last_scalar_replays() const;\n#endif" header "${header}")
string(REPLACE "std::size_t ScalarPathEvaluator::last_contributing_paths() const { return impl_->contributions; }"
  "std::size_t ScalarPathEvaluator::last_contributing_paths() const { return impl_->contributions; }\n#ifdef XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY\n  std::size_t ScalarPathEvaluator::last_scalar_replays() const { return impl_->scalar_replayed_lanes; }\n#endif" source "${source}")
string(REPLACE "x.contributions   = 0;"
  "x.contributions   = 0;\n#ifdef XCA_EXPERIMENT_SCALAR_CUTOFF_REPLAY\n    x.scalar_replayed_lanes = 0;\n#endif" source "${source}")
set(source "#include <map>\n${source}")
set(source "#ifdef XCA_EXPERIMENT_SCALAR_AUDIT\n#include \"audit.hpp\"\n#endif\n${source}")
set(source "#ifdef XCA_EXPERIMENT_SCALAR_RESIDENT\n#include \"resident_kernels.hpp\"\n#endif\n${source}")
string(REPLACE "kernel_bytes = checked_multiply(2, tensor_bytes(1, r, poles.size()));"
  "kernel_bytes = checked_multiply(2, tensor_bytes(1, r, poles.size()));\n#ifdef XCA_EXPERIMENT_SCALAR_RESIDENT\n      if (scalar_eligible()) kernel_bytes = checked_add(kernel_bytes, checked_multiply(2, tensor_bytes(r, model.num_fields(), model.num_fields())));\n#endif" source "${source}")
string(REPLACE "negative.emplace_back(minus);\n      }\n      ops->wait_for_default_uploads();"
  "negative.emplace_back(minus);\n      }\n#ifdef XCA_EXPERIMENT_SCALAR_RESIDENT\n      if (scalar_eligible()) prepare_resident_weights();\n#endif\n      ops->wait_for_default_uploads();" source "${source}")
string(REPLACE "\"diagram_evaluator.hpp\"" "\"scalar_evaluator.hpp\"\n#include \"../scalar_edges/scalar_pack.hpp\"" source "${source}")
string(REPLACE "    void evaluate(std::vector<Tensor>" "#include \"scheduler.inc\"\n    void evaluate(std::vector<Tensor>" source "${source}")
string(REPLACE "if (first != last) x.prepare_self_energy(topology);" "if (first != last) x.prepare_self_energy(topology,x.scalar_extra_bound());" source "${source}")
set(original "        x.counts = for_each_self_energy_path(x.model, x.hyb->poles, topology, live, first, last,
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
string(REPLACE "self_energy_storage_bound() const { return impl_->self_energy_bound(false); }"
  "self_energy_storage_bound() const { return checked_add(impl_->self_energy_bound(false),impl_->scalar_extra_bound()); }" source "${source}")
string(REPLACE "    std::vector<Tensor> input(G.size()), output(G.size());"
  "#ifdef XCA_EXPERIMENT_SCALAR_ARENA\n    if(first!=last && x.scalar_eligible())return x.scalar_arena_compute(G,topology,live,first,last);\n#endif\n    std::vector<Tensor> input(G.size()), output(G.size());" source "${source}")
string(REPLACE "return impl_->self_energy_bound(topology.extent(0) == 1);" "return checked_add(impl_->self_energy_bound(topology.extent(0) == 1),impl_->scalar_extra_bound());" source "${source}")
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
  add_executable(block_sparse_scalar_path_tests "${CMAKE_CURRENT_BINARY_DIR}/scalar_path_tests.cpp" qualification.cpp)
  target_link_libraries(block_sparse_scalar_path_tests PRIVATE block_sparse_scalar_path_evaluator ${PROJECT_NAME}_test_utils gtest_main)
  add_test(NAME "experiments/block_sparse_cuda/scalar_paths" COMMAND block_sparse_scalar_path_tests)
endif()

add_library(block_sparse_scalar_fanin_evaluator STATIC "${CMAKE_CURRENT_BINARY_DIR}/scalar_evaluator.cpp")
target_include_directories(block_sparse_scalar_fanin_evaluator PUBLIC "${CMAKE_CURRENT_BINARY_DIR}" "${source_dir}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(block_sparse_scalar_fanin_evaluator PRIVATE XCA_EXPERIMENT_SCALAR_FANIN)
target_link_libraries(block_sparse_scalar_fanin_evaluator PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c block_sparse_scalar_edge_kernels)
xca_add_benchmark(block_sparse_scalar_fanin_benchmark)
target_link_libraries(block_sparse_scalar_fanin_benchmark PRIVATE block_sparse_scalar_fanin_evaluator)
if(Build_Tests)
  add_executable(block_sparse_scalar_fanin_tests "${CMAKE_CURRENT_BINARY_DIR}/scalar_path_tests.cpp" qualification.cpp fanin_order.cpp)
  target_link_libraries(block_sparse_scalar_fanin_tests PRIVATE block_sparse_scalar_fanin_evaluator ${PROJECT_NAME}_test_utils gtest_main)
  add_test(NAME "experiments/block_sparse_cuda/scalar_fanin" COMMAND block_sparse_scalar_fanin_tests)
endif()
xca_add_benchmark(profile_fanin)
target_link_libraries(profile_fanin PRIVATE block_sparse_scalar_fanin_evaluator)

add_library(block_sparse_scalar_arena_evaluator STATIC "${CMAKE_CURRENT_BINARY_DIR}/scalar_evaluator.cpp")
target_include_directories(block_sparse_scalar_arena_evaluator PUBLIC "${CMAKE_CURRENT_BINARY_DIR}" "${source_dir}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(block_sparse_scalar_arena_evaluator PRIVATE XCA_EXPERIMENT_SCALAR_FANIN XCA_EXPERIMENT_SCALAR_ARENA)
target_link_libraries(block_sparse_scalar_arena_evaluator PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c block_sparse_scalar_edge_kernels)
xca_add_benchmark(block_sparse_scalar_arena_benchmark)
target_link_libraries(block_sparse_scalar_arena_benchmark PRIVATE block_sparse_scalar_arena_evaluator)
if(Build_Tests)
  add_executable(block_sparse_scalar_arena_tests "${CMAKE_CURRENT_BINARY_DIR}/scalar_path_tests.cpp" qualification.cpp fanin_order.cpp)
  target_link_libraries(block_sparse_scalar_arena_tests PRIVATE block_sparse_scalar_arena_evaluator ${PROJECT_NAME}_test_utils gtest_main)
  add_test(NAME "experiments/block_sparse_cuda/scalar_arena" COMMAND block_sparse_scalar_arena_tests)
endif()
add_executable(profile_arena profile_fanin.cpp)
target_link_libraries(profile_arena PRIVATE block_sparse_scalar_arena_evaluator ${PROJECT_NAME}_warnings)

add_library(block_sparse_scalar_constants_evaluator STATIC "${CMAKE_CURRENT_BINARY_DIR}/scalar_evaluator.cpp")
target_include_directories(block_sparse_scalar_constants_evaluator PUBLIC "${CMAKE_CURRENT_BINARY_DIR}" "${source_dir}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(block_sparse_scalar_constants_evaluator PRIVATE XCA_EXPERIMENT_SCALAR_FANIN XCA_EXPERIMENT_SCALAR_ARENA XCA_EXPERIMENT_SCALAR_CONSTANTS)
target_link_libraries(block_sparse_scalar_constants_evaluator PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c block_sparse_scalar_edge_kernels)
add_executable(block_sparse_scalar_constants_benchmark block_sparse_scalar_arena_benchmark.cpp)
target_compile_definitions(block_sparse_scalar_constants_benchmark PRIVATE XCA_EXPERIMENT_SCALAR_CONSTANTS)
target_link_libraries(block_sparse_scalar_constants_benchmark PRIVATE block_sparse_scalar_constants_evaluator ${PROJECT_NAME}_warnings)
if(Build_Tests)
  add_executable(block_sparse_scalar_constants_tests "${CMAKE_CURRENT_BINARY_DIR}/scalar_path_tests.cpp" qualification.cpp fanin_order.cpp constants_layout.cpp)
  target_link_libraries(block_sparse_scalar_constants_tests PRIVATE block_sparse_scalar_constants_evaluator ${PROJECT_NAME}_test_utils gtest_main)
  add_test(NAME "experiments/block_sparse_cuda/scalar_constants" COMMAND block_sparse_scalar_constants_tests)
endif()

cmake_minimum_required(VERSION 3.22)
project(xca_fast_convolve_benchmark LANGUAGES CXX)

find_package(TRIQS 4.0 REQUIRED)

add_executable(fast_convolve_benchmark main.cpp)
target_include_directories(fast_convolve_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../c++)
target_link_libraries(fast_convolve_benchmark PRIVATE triqs nda::blas_lapack)

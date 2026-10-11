#pragma once
#include <cuComplex.h>
#include <cuda_runtime_api.h>
extern "C" cudaError_t xca_solve_weighted_add(cuDoubleComplex const *,
                                              cuDoubleComplex const *,
                                              cuDoubleComplex const *,
                                              cuDoubleComplex *, int, int,
                                              cudaStream_t);
extern "C" cudaError_t xca_solve_add_scale(cuDoubleComplex const *,
                                           cuDoubleComplex *, int, double,
                                           cudaStream_t);
extern "C" cudaError_t xca_solve_product(cuDoubleComplex const *,
                                         cuDoubleComplex const *,
                                         cuDoubleComplex *, int,
                                         cuDoubleComplex, cuDoubleComplex,
                                         cudaStream_t);
extern "C" cudaError_t xca_solve_vertex(cuDoubleComplex const *const *,
                                        cuDoubleComplex const *const *,
                                        cuDoubleComplex *const *, int,
                                        cuDoubleComplex, cuDoubleComplex,
                                        cudaStream_t);

extern "C" cudaError_t xca_solve_lu(cuDoubleComplex const *, int const *,
                                    cuDoubleComplex const *, cuDoubleComplex *,
                                    int, int, cudaStream_t);

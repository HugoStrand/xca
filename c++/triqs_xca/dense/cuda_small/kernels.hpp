#pragma once
#include <cuComplex.h>
#include <cuda_runtime_api.h>
extern "C" cudaError_t xca_small_weighted_add(cuDoubleComplex const *, cuDoubleComplex const *, cuDoubleComplex const *, cuDoubleComplex *, int, int,
                                              cudaStream_t);
extern "C" cudaError_t xca_small_add_scale(cuDoubleComplex const *, cuDoubleComplex *, int, double, cudaStream_t);
extern "C" cudaError_t xca_small_product(cuDoubleComplex const *, cuDoubleComplex const *, cuDoubleComplex *, int, int, cuDoubleComplex, cuDoubleComplex,
                                         cudaStream_t);
extern "C" cudaError_t xca_small_vertex(cuDoubleComplex const *const *, cuDoubleComplex const *const *, cuDoubleComplex *const *, int, int,
                                        cuDoubleComplex, cuDoubleComplex, cudaStream_t);

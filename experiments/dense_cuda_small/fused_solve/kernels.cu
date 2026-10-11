#include "kernels.hpp"
namespace {
// Each operation retains the original sequential complex multiply/add stages.
// Preserve library operand order under the normal CUDA FMA contraction.
__global__ void weighted_add(cuDoubleComplex const *in,
                             cuDoubleComplex const *w,
                             cuDoubleComplex const *parity,
                             cuDoubleComplex *out, int count, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  auto value = cuCmul(in[i], w[i / stride]);
  value = cuCmul(value, parity[i / stride]);
  out[i] = cuCadd(out[i], value);
}
__global__ void add_scale(cuDoubleComplex const *in, cuDoubleComplex *out,
                          int count, double scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  out[i] = cuCmul(make_cuDoubleComplex(scale, 0.), cuCadd(out[i], in[i]));
}
__device__ cuDoubleComplex product_element(cuDoubleComplex const *a,
                                           cuDoubleComplex const *b, int i,
                                           int j) {
  double re = 0., im = 0.;
  // Probe grouped real dot products rather than interleaved complex products.
  // All public API parity gates remain required; no weaker GEMM tolerance.
#pragma unroll
  for (int k = 0; k < 4; ++k)
    re = __fma_rn(cuCreal(a[i * 4 + k]), cuCreal(b[k * 4 + j]), re);
#pragma unroll
  for (int k = 0; k < 4; ++k)
    re = __fma_rn(-cuCimag(a[i * 4 + k]), cuCimag(b[k * 4 + j]), re);
#pragma unroll
  for (int k = 0; k < 4; ++k)
    im = __fma_rn(cuCreal(a[i * 4 + k]), cuCimag(b[k * 4 + j]), im);
#pragma unroll
  for (int k = 0; k < 4; ++k)
    im = __fma_rn(cuCimag(a[i * 4 + k]), cuCreal(b[k * 4 + j]), im);
  return make_cuDoubleComplex(re, im);
}
__device__ void store(cuDoubleComplex *out, int element, cuDoubleComplex value,
                      cuDoubleComplex alpha, cuDoubleComplex beta) {
  value = cuCmul(alpha, value);
  if (cuCreal(beta) != 0. || cuCimag(beta) != 0.)
    value = cuCadd(value, cuCmul(beta, out[element]));
  out[element] = value;
}
__global__ void product(cuDoubleComplex const *a, cuDoubleComplex const *b,
                        cuDoubleComplex *out, int batches,
                        cuDoubleComplex alpha, cuDoubleComplex beta) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= batches * 16)
    return;
  int batch = q / 16, element = q % 16;
  auto value =
      product_element(a + batch * 16, b + batch * 16, element / 4, element % 4);
  store(out + batch * 16, element, value, alpha, beta);
}
__global__ void vertex(cuDoubleComplex const *const *a,
                       cuDoubleComplex const *const *b,
                       cuDoubleComplex *const *out, int batches,
                       cuDoubleComplex alpha, cuDoubleComplex beta) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= batches * 16)
    return;
  int batch = q / 16, element = q % 16;
  auto value = product_element(a[batch], b[batch], element / 4, element % 4);
  store(out[batch], element, value, alpha, beta);
}
// F=LU factorization is column-major when viewing the cached C-order table.
// OP_T solve: U^T forward, L^T unit-diagonal backward, reverse pivots.
__global__ void transpose_solve(cuDoubleComplex const *lu, int const *pivots,
                                cuDoubleComplex const *values,
                                cuDoubleComplex *out, int rank, int columns) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= columns)
    return;
  cuDoubleComplex x[40];
  for (int i = 0; i < rank; ++i)
    x[i] = values[i * columns + column];
  for (int i = 0; i < rank; ++i) {
    auto value = x[i];
    for (int k = 0; k < i; ++k)
      value = cuCsub(value, cuCmul(lu[i * rank + k], x[k]));
    x[i] = cuCdiv(value, lu[i * rank + i]);
  }
  for (int i = rank - 1; i >= 0; --i) {
    auto value = x[i];
    for (int k = i + 1; k < rank; ++k)
      value = cuCsub(value, cuCmul(lu[i * rank + k], x[k]));
    x[i] = value;
  }
  for (int i = rank - 1; i >= 0; --i) {
    auto value = x[i];
    x[i] = x[pivots[i] - 1];
    x[pivots[i] - 1] = value;
  }
  for (int i = 0; i < rank; ++i)
    out[i * columns + column] = x[i];
}
} // namespace
extern "C" cudaError_t xca_solve_weighted_add(cuDoubleComplex const *in,
                                              cuDoubleComplex const *w,
                                              cuDoubleComplex const *parity,
                                              cuDoubleComplex *out, int count,
                                              int stride, cudaStream_t stream) {
  weighted_add<<<(count + 127) / 128, 128, 0, stream>>>(in, w, parity, out,
                                                        count, stride);
  return cudaGetLastError();
}
extern "C" cudaError_t xca_solve_add_scale(cuDoubleComplex const *in,
                                           cuDoubleComplex *out, int count,
                                           double scale, cudaStream_t stream) {
  add_scale<<<(count + 127) / 128, 128, 0, stream>>>(in, out, count, scale);
  return cudaGetLastError();
}
extern "C" cudaError_t
xca_solve_product(cuDoubleComplex const *a, cuDoubleComplex const *b,
                  cuDoubleComplex *out, int batches, cuDoubleComplex alpha,
                  cuDoubleComplex beta, cudaStream_t stream) {
  product<<<(batches * 16 + 127) / 128, 128, 0, stream>>>(a, b, out, batches,
                                                          alpha, beta);
  return cudaGetLastError();
}
extern "C" cudaError_t xca_solve_vertex(cuDoubleComplex const *const *a,
                                        cuDoubleComplex const *const *b,
                                        cuDoubleComplex *const *out,
                                        int batches, cuDoubleComplex alpha,
                                        cuDoubleComplex beta,
                                        cudaStream_t stream) {
  vertex<<<(batches * 16 + 127) / 128, 128, 0, stream>>>(a, b, out, batches,
                                                         alpha, beta);
  return cudaGetLastError();
}

extern "C" cudaError_t xca_solve_lu(cuDoubleComplex const *lu,
                                    int const *pivots,
                                    cuDoubleComplex const *values,
                                    cuDoubleComplex *out, int rank, int columns,
                                    cudaStream_t stream) {
  transpose_solve<<<(columns + 127) / 128, 128, 0, stream>>>(
      lu, pivots, values, out, rank, columns);
  return cudaGetLastError();
}

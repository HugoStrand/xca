#include "kernels.hpp"
namespace {
  // Each operation retains the original sequential complex multiply/add stages.
  // Preserve library operand order under the normal CUDA FMA contraction.
  __global__ void weighted_add(cuDoubleComplex const *in, cuDoubleComplex const *w, cuDoubleComplex const *parity, cuDoubleComplex *out, int count,
                               int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    auto value = cuCmul(in[i], w[i / stride]);
    value      = cuCmul(value, parity[i / stride]);
    out[i]     = cuCadd(out[i], value);
  }
  __global__ void add_scale(cuDoubleComplex const *in, cuDoubleComplex *out, int count, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = cuCmul(make_cuDoubleComplex(scale, 0.), cuCadd(out[i], in[i]));
  }
  template<int N> __device__ cuDoubleComplex product_element(cuDoubleComplex const *a, cuDoubleComplex const *b, int i, int j) {
    double re = 0., im = 0.;
    // Probe grouped real dot products rather than interleaved complex products.
    // All public API parity gates remain required; no weaker GEMM tolerance.
#pragma unroll
    for (int k = 0; k < N; ++k) re = __fma_rn(cuCreal(a[i * N + k]), cuCreal(b[k * N + j]), re);
#pragma unroll
    for (int k = 0; k < N; ++k) re = __fma_rn(-cuCimag(a[i * N + k]), cuCimag(b[k * N + j]), re);
#pragma unroll
    for (int k = 0; k < N; ++k) im = __fma_rn(cuCreal(a[i * N + k]), cuCimag(b[k * N + j]), im);
#pragma unroll
    for (int k = 0; k < N; ++k) im = __fma_rn(cuCimag(a[i * N + k]), cuCreal(b[k * N + j]), im);
    return make_cuDoubleComplex(re, im);
  }
  __device__ void store(cuDoubleComplex *out, int element, cuDoubleComplex value, cuDoubleComplex alpha, cuDoubleComplex beta) {
    value = cuCmul(alpha, value);
    if (cuCreal(beta) != 0. || cuCimag(beta) != 0.) value = cuCadd(value, cuCmul(beta, out[element]));
    out[element] = value;
  }
  template<int N> __global__ void product(cuDoubleComplex const *a, cuDoubleComplex const *b, cuDoubleComplex *out, int batches, cuDoubleComplex alpha,
                          cuDoubleComplex beta) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batches * (N * N)) return;
    int batch = q / (N * N), element = q % (N * N);
    auto value = product_element<N>(a + batch * (N * N), b + batch * (N * N), element / N, element % N);
    store(out + batch * (N * N), element, value, alpha, beta);
  }
  template<int N> __global__ void vertex(cuDoubleComplex const *const *a, cuDoubleComplex const *const *b, cuDoubleComplex *const *out, int batches,
                         cuDoubleComplex alpha, cuDoubleComplex beta) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batches * (N * N)) return;
    int batch = q / (N * N), element = q % (N * N);
    auto value = product_element<N>(a[batch], b[batch], element / N, element % N);
    store(out[batch], element, value, alpha, beta);
  }
} // namespace
extern "C" cudaError_t xca_pow2_weighted_add(cuDoubleComplex const *in, cuDoubleComplex const *w, cuDoubleComplex const *parity,
                                              cuDoubleComplex *out, int count, int stride, cudaStream_t stream) {
  weighted_add<<<(count + 127) / 128, 128, 0, stream>>>(in, w, parity, out, count, stride);
  return cudaGetLastError();
}
extern "C" cudaError_t xca_pow2_add_scale(cuDoubleComplex const *in, cuDoubleComplex *out, int count, double scale, cudaStream_t stream) {
  add_scale<<<(count + 127) / 128, 128, 0, stream>>>(in, out, count, scale);
  return cudaGetLastError();
}
extern "C" cudaError_t xca_pow2_product(cuDoubleComplex const *a, cuDoubleComplex const *b, cuDoubleComplex *out, int batches, int dimension, cuDoubleComplex alpha,
                                         cuDoubleComplex beta, cudaStream_t stream) {
  if (batches == 0) return cudaSuccess;
  if (batches < 0) return cudaErrorInvalidValue;
  switch (dimension) {
  case 2: product<2><<<(batches * 4 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 4: product<4><<<(batches * 16 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 8: product<8><<<(batches * 64 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 16: product<16><<<(batches * 256 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 32: product<32><<<(batches * 1024 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 64: product<64><<<(batches * 4096 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
extern "C" cudaError_t xca_pow2_vertex(cuDoubleComplex const *const *a, cuDoubleComplex const *const *b, cuDoubleComplex *const *out, int batches, int dimension,
                                        cuDoubleComplex alpha, cuDoubleComplex beta, cudaStream_t stream) {
  if (batches == 0) return cudaSuccess;
  if (batches < 0) return cudaErrorInvalidValue;
  switch (dimension) {
  case 2: vertex<2><<<(batches * 4 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 4: vertex<4><<<(batches * 16 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 8: vertex<8><<<(batches * 64 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 16: vertex<16><<<(batches * 256 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 32: vertex<32><<<(batches * 1024 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  case 64: vertex<64><<<(batches * 4096 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta); break;
  default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

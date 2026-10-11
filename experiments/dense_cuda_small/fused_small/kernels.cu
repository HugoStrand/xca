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
  __device__ cuDoubleComplex product_element(cuDoubleComplex const *a, cuDoubleComplex const *b, int i, int j) {
    double re = 0., im = 0.;
    // Probe grouped real dot products rather than interleaved complex products.
    // All public API parity gates remain required; no weaker GEMM tolerance.
#pragma unroll
    for (int k = 0; k < 4; ++k) re = __fma_rn(cuCreal(a[i * 4 + k]), cuCreal(b[k * 4 + j]), re);
#pragma unroll
    for (int k = 0; k < 4; ++k) re = __fma_rn(-cuCimag(a[i * 4 + k]), cuCimag(b[k * 4 + j]), re);
#pragma unroll
    for (int k = 0; k < 4; ++k) im = __fma_rn(cuCreal(a[i * 4 + k]), cuCimag(b[k * 4 + j]), im);
#pragma unroll
    for (int k = 0; k < 4; ++k) im = __fma_rn(cuCimag(a[i * 4 + k]), cuCreal(b[k * 4 + j]), im);
    return make_cuDoubleComplex(re, im);
  }
  __device__ void store(cuDoubleComplex *out, int element, cuDoubleComplex value, cuDoubleComplex alpha, cuDoubleComplex beta) {
    value = cuCmul(alpha, value);
    if (cuCreal(beta) != 0. || cuCimag(beta) != 0.) value = cuCadd(value, cuCmul(beta, out[element]));
    out[element] = value;
  }
  __global__ void product(cuDoubleComplex const *a, cuDoubleComplex const *b, cuDoubleComplex *out, int batches, cuDoubleComplex alpha,
                          cuDoubleComplex beta) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batches * 16) return;
    int batch = q / 16, element = q % 16;
    auto value = product_element(a + batch * 16, b + batch * 16, element / 4, element % 4);
    store(out + batch * 16, element, value, alpha, beta);
  }
  __global__ void vertex(cuDoubleComplex const *const *a, cuDoubleComplex const *const *b, cuDoubleComplex *const *out, int batches,
                         cuDoubleComplex alpha, cuDoubleComplex beta) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batches * 16) return;
    int batch = q / 16, element = q % 16;
    auto value = product_element(a[batch], b[batch], element / 4, element % 4);
    store(out[batch], element, value, alpha, beta);
  }
} // namespace
extern "C" cudaError_t xca_fused_weighted_add(cuDoubleComplex const *in, cuDoubleComplex const *w, cuDoubleComplex const *parity,
                                              cuDoubleComplex *out, int count, int stride, cudaStream_t stream) {
  weighted_add<<<(count + 127) / 128, 128, 0, stream>>>(in, w, parity, out, count, stride);
  return cudaGetLastError();
}
extern "C" cudaError_t xca_fused_add_scale(cuDoubleComplex const *in, cuDoubleComplex *out, int count, double scale, cudaStream_t stream) {
  add_scale<<<(count + 127) / 128, 128, 0, stream>>>(in, out, count, scale);
  return cudaGetLastError();
}
extern "C" cudaError_t xca_fused_product(cuDoubleComplex const *a, cuDoubleComplex const *b, cuDoubleComplex *out, int batches, cuDoubleComplex alpha,
                                         cuDoubleComplex beta, cudaStream_t stream) {
  product<<<(batches * 16 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta);
  return cudaGetLastError();
}
extern "C" cudaError_t xca_fused_vertex(cuDoubleComplex const *const *a, cuDoubleComplex const *const *b, cuDoubleComplex *const *out, int batches,
                                        cuDoubleComplex alpha, cuDoubleComplex beta, cudaStream_t stream) {
  vertex<<<(batches * 16 + 127) / 128, 128, 0, stream>>>(a, b, out, batches, alpha, beta);
  return cudaGetLastError();
}

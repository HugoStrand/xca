# E4: pointwise fusion and N4 products

Isolated ordinary-SE prototype derived from E3. Production remains unchanged.
XCA_FUSED_MODE=0 retains library operations, mode1 (default) fuses special-vertex
weighting, separate parity multiplication and accumulation, plus final convolution
addition/scaling. Multiplication/addition stages retain order; custom kernels compile
with ordinary CUDA FMA contraction and the same input-first weighting order as
cuBLAS DGMM. Reversing complex multiply operands or disabling contraction failed
strict high-order checks; those controls are preserved in results/validation/.
Cached library LU and rank GEMMs are retained.

Mode2 additionally probes custom complex-double N4 products/vertex products;
non-N4 products use libraries. Interleaved complex-dot rounding failed strict high-order checks; the grouped real
FMA version passes all six CTest entries, including long-index/order12 and tails.
Broader benchmark gates pass:17 runs/250 backend-case groups, sampled errors zero. No tolerance is weakened.

XCA_BATCH_CAPACITY=1 (default) measures fusion independently of cross-index width;
8/32/128 probe useful E3+E4 combinations. E3 metadata cache, lane tails, sequential
output accumulation and upload-event handoff are retained. Each kernel uses128
threads with bounds checks; the N4 probe uses one thread per output element, four
grouped real FMA dot products, and no mixed precision. No claimed occupancy gain yet.

Build from the repository with CUDA environment and clang++-19 host CXX; the separate
CUDA target uses nvcc at architecture86/C++17 and has no NDA headers. Standalone
CMake targets mirror E3 and add fused_primitives. Mode1 passes six CTest entries,
including strict high-order API checks and exact pointwise primitive parity.

```bash
source ~/env/xca-cuda.sh
cmake -S experiments/dense_cuda_small/fused_small -B cbuild-experiment-fused \
  -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Release
cmake --build cbuild-experiment-fused --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild-experiment-fused --parallel 1 --output-on-failure
XCA_FUSED_MODE=1 XCA_BATCH_CAPACITY=128 \
  python3 benchmarks/dense_cuda_small/run.py \
  --binary cbuild-experiment-fused/fused_small_benchmark \
  --candidate-sources experiments/dense_cuda_small \
  --output cbuild-small-runs/my-fused-run --profile throughput -- --methods se
```

Timings, strict gates, sanitizer checks and traversal audits pass. See RESULTS.md
and ../COMPARISON.md for capacity selection, cold-context costs and limitations.

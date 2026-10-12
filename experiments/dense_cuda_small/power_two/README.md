# Power-of-two dense kernels

M1 experiment derived from retained `fused_small`, with its own namespace and C ABI.
Only ordinary self-energy is native. Test adapters delegate other methods to the
baseline; those methods do not validate new kernels. All original folders remain.

Mode 2 compiles grouped real-FMA products and pointer-table vertices for matrix
sizes 2, 4, 8, 16, 32 and 64. Mode 0 is the library control; mode 1 retains pointwise
fusion with library products. These are experimental candidates, not a qualified
production dispatch table. Rank transforms and pivoted LU stay in libraries.
Prototype resource limits remain until M2. No fast math or tolerance relaxation.

Configure with the CUDA environment and an explicit architecture for the test GPU:

```bash
source ~/env/xca-cuda.sh
cmake -S experiments/dense_cuda_small/power_two -B cbuild-experiment-power-two \
  -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cbuild-experiment-power-two --parallel 6
XCA_FUSED_MODE=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild-experiment-power-two --parallel 1 --output-on-failure
XCA_FUSED_MODE=2 XCA_BATCH_CAPACITY=128 \
  python3 benchmarks/dense_cuda_small/run.py \
  --binary cbuild-experiment-power-two/power_two_benchmark \
  --candidate-sources experiments/dense_cuda_small/power_two \
  --output cbuild-small-runs/power-two-n8 --profile quick -- \
  --dimension 8 --methods se
```

The existing physical harness covers dimensions 2–16. Primitive tests cover 2–64;
larger physical fixtures and crossover measurements remain to be added. A4000
measurements are local controls; GH200-class cards are the optimization target and
require independent qualification before choosing production thresholds.

`XCA_FUSED_MODE=2 cbuild-experiment-power-two/primitive_benchmark` emits device-event
latencies for products and left pointer-table vertices at dimensions 2–64 and lane
counts 1/8/32/128, rank 16. Each case checks finite results and the strict relative
gate before timing 100 operations after ten warmups. This measures primitive time,
not evaluator time; do not infer a production cutoff from it. Repeat in separate
processes and retain source/binary hashes for decisions. Primitive tests additionally
cover both vertex directions, complex alpha, and beta=0/1.

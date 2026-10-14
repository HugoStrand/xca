# NCA-specific admission

Target: GH200-class hardware. This private E7 control retains production arithmetic,
operator/special policies and all prior experiments. It changes self-energy admission
for a validated one-line topology, where no DLR solve/convolution occurs.

The no-argument bound remains the generic upper envelope. An added topology overload
subtracts the six inactive DLR scratch tiles for NCA, retaining fixed rank/LU tables,
two complete functions, seven caller tiles, operator reserve, weights and mapped
metadata. Higher orders and correlators retain their original bounds. The candidate
uses the topology bound for actual admission before input/output upload.

Budget rejection precedes state changes. If an NCA call can fit its explicit budget,
completed scratch from a prior higher-order call is released before querying free
memory/headroom. Release synchronizes the owned stream and discards RHS/convolution
buffers, preserving tables, solver and shared handles. It is idempotent. Thus cached
scratch cannot invalidate the smaller envelope or survive as an uncounted allocation.
Device failures propagate; no CPU replay or arithmetic fallback is introduced.

Ten tests reuse eight current observable/API controls and add tight-budget NCA
acceptance, bound-minus-one rejection before work, empty ranges, higher-order rejection
and recovery, OCA/NCA/OCA transitions, and independent scratch release/reconstruction.
Reference and mapped policies are exercised when available. Results agree exactly
with the production evaluator in the explicit transition cases. Two focused cases
also receive memcheck/full leak checking. No production/Python contract changes yet.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DBuild_Benchmarks=ON -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_nca_admission_tests --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild-cuda \
 -R '^experiments/block_sparse_cuda/nca_admission$' --output-on-failure
```

This qualifies budget/lifetime behavior, not runtime speedup. The smaller declaration
removes an inactive reserve; it does not save convolution arithmetic or shrink complete
inputs/outputs. Host/opaque allocation peaks and fourteen-mode full calls remain
unqualified. CPU-only builds omit the CUDA fixture; native compilation is optional
for the reference policy. Promote only after extending production bound accessors,
updating budget tests against the actual topology envelope, preserving extra reserves
in private scalar schedulers, and requalifying wrappers/MPI. Keep the generic bound
available for callers that need an order-independent upper estimate.

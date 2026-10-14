# DLR convolution in independent row chunks

Target: GH200-class accelerators. Local RTX A4000 results are development evidence.
This private experiment leaves production DlrOps and every earlier experiment intact.
It generates RowDlrOps from the current reference, retaining the original full method,
and adds an explicitly selected convolution over output-row chunks.

The inner matrix dimension K is never tiled. Each chunk packs contiguous coefficient
rows, applies the same H/Q/C tables and ordered three-product formula, then scatters
completed rows. The full transformed right input remains resident. Solves are unchanged;
completed RHS/old convolution scratch is released before allocating row workspace.
Output must be disjoint from both inputs. Preflight rejects invalid shapes, aliases
and budgets before release or output writes. Submitted failures propagate with stream
cleanup, without replay. Narrower BLAS calls may change rounding even with unchanged
K reduction length; this is not a bitwise-equivalence or cutoff-safety claim.

For B=min(rows,M), the explicit workspace bound includes the existing five rank tables,
pivots/status, plus `r*(K*N + 2*B*K + 3*B*N)` complex entries. Caller coefficient inputs
and complete output are additional. Temporary row storage is freed at call completion;
`storage_bytes()` reports retained tables afterward. Opaque library allocations and
allocator granularity are outside that bound. Large-sector estimates do not qualify
a complete N=14 atom_diag or evaluator run.

With CUDA TRIQS and Build_Benchmarks=ON:

```sh
cmake --build cbuild-cuda --parallel 6 --target block_sparse_row_dlr_tests block_sparse_row_dlr_benchmark
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild-cuda -R '^experiments/block_sparse_cuda/convolution_rows$' --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cbuild-cuda/benchmarks/experiments/block_sparse_cuda/convolution_rows/block_sparse_row_dlr_benchmark --m 210 --k 210 --n 252 --rows 16 --samples 3
```

Tests cover rectangular shapes/tails, multiple ranks, CPU/reference comparisons,
retained solve scratch release, exact-budget admission, aliases/recovery and a
threshold-neighborhood diagnostic. Numerical screens are explicitly
`1e-13*reference_norm + 1e-14`; the threshold diagnostic uses a relative bound and
records keep/drop decisions rather than declaring them universally invariant.
Benchmark calls include packing, allocation, execution, synchronization and scatter.
Reference scratch is warm/reused; row temporaries are allocated per chunk, so setup
and allocator overhead are deliberately included. Both routes must repeat their own
warm-up result exactly and pass finite/CPU/reference screens.

The CLI caps dimensions at 512, samples at five and lambda at 120. These are bounded
primitive replays, not Solver/topology-sum timings. Full-call integration, admission
liveness, wider cancellation/cutoff coverage, alias fallback and target performance
remain separate gates before any production decision. Generic methodology stays here;
results and hardware-specific comments belong in informative subdirectories.

## Complete evaluator experiment

`RowDiagramEvaluator` is generated from the current production source with checked
replacement seams. Its additional last constructor argument selects a positive row
cap (default eight). It routes both self-energy edge directions and correlator
fragment edges through the row primitive. All production APIs remain intact in this
private class, including complete host output placement and NCA fallback. The declared
DLR reserve is tables + full maximum rank tensor + five maximum row strips; the
seven caller rank tensors and input/output reserves are unchanged. NCA estimates
omit this inactive convolution scratch as in production. No Python integration or
production promotion is implied.

Build `block_sparse_row_evaluator_tests`, `block_sparse_row_evaluator_benchmark` and
`block_sparse_row_evaluator_cutoff_audit`. Inherited production controls plus additional
tests cover changed inputs/masks, output placement, exact budget admission/rejection,
repeated calls and recovery. The benchmark uses the shared complete-call fixture and
the library sparse/reference special-vertex policies. Set `XCA_EXPERIMENT_CONV_ROWS`
to a cap between 1 and 512. Setting `XCA_EXPERIMENT_SAMPLE_ROWS` additionally samples
free device memory while row temporaries are live; measure this diagnostic separately
from normal latency. It is a sampled global free-memory delta, not a continuous or
process-isolated peak. Library allocations and other activity can affect it.

The cutoff audit bisects an active physical OCA call to adjacent representable
propagator scales, then compares nine neighboring scales at two ranks and three row
caps. Its CSV includes contribution counts, errors and explicit decision agreement.
A successful process means the diagnostic completed, **not** that its candidate passed
semantic acceptance. At rank23/cap1 one sampled scale retains a contribution in the
reference and drops it in the candidate. Keep the route experimental; a tolerance
comparison at ordinary scales does not resolve this disagreement. Preserve the
counterexample rather than modifying the production cutoff.

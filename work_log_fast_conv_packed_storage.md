# Fast convolution and packed storage work log

## Saved numerical baseline and implementation plan — 2026-11-03

Source: `59c7bbd` `[cppdlr] reuse convolution storage`.

Committed the standalone complex time-ordered fast-convolution helper, exact-
shape reusable workspace, direct right-side LU solves, output GEMMs, and six
focused tests. The cppdlr extension links NDA's exported BLAS target for TRSM.
The focused CTest passed before committing (0.18 s). The helper was also
validated in the previously recorded scalar and sparse-layout experiments.

`plan_fast_conv_packed_storage.md` preserves the measurement findings, their
scope, ownership and active-shape requirements, and staged implementation.
Sparse API/helper refactoring must finish first. Dense fast integration and
CPU batching proceed independently, initially keeping convolution ownership
in the dense evaluator. Mutable scratch will always remain evaluator-owned.

The selected numerical segment speedups are 1.28--1.58x for physical test
models; the maximum observed relative error was 4.7e-14. Packing alone is a
small contribution; fast convolution supplies most of the benefit. These are
not complete solver timings. Raw results and reproducibility instructions
are under `benchmarks/block_sparse_layout/` and `benchmarks/fast_convolve/`.

Existing user edits include constructor formatting in the dense evaluator and
unrelated .gitignore changes. They are preserved and excluded from source
commits. A snapshot of the dense formatting patch is at
`/tmp/xca-dense-user-formatting-20261103.patch`; temporary files are not the
durable work record. No GPU allocation or network push is needed.

## Scalar dense integration — 2026-11-03

Source: `[dense] use fast convolution` (preceding source checkpoint).

Both dense edge convolution directions now use the output-parameter helper
and one reusable exact-shape workspace. The evaluator owns both for this
intermediate stage; Action remains unchanged. Existing free numerical helpers
receive these dependencies explicitly. Output aliases the running input after
coefficient conversion, eliminating temporary convolution result assignment.

`dense/prepared_evaluator` passed after rebuilding. Its new lifetime case
copies the evaluator, destroys the original and its model, and then evaluates
again. The mesh shares the heap-owned interpolation object, so the helper's
borrow remains valid. No reference numerical tolerances changed. Broader
regression testing follows the concurrent sparse API/batching stages.

The user's constructor-formatting patch was explicitly reversed in the index
before committing this source change; the worktree preserves it. Benchmark
code and existing measurements are separately committed as `c5a596b`.

## Follow-up goal started — 2026-11-03

The prerequisite sparse structural refactoring passed all 85 CPU CTest entries
in 33.46 s at `db2e2fc`. Its persistent goal is complete; the separate goal to
implement this plan is active. The scalar and batched dense evaluators are
already validated with evaluator-owned convolution helpers/workspaces.

Before changing sparse numerical storage, capture matched prepared-call
baselines including path discovery, endpoint sums, correlator completion,
result ownership and reset. Use bounded samples from physical partitions
with maximum dimensions one, two and six, at DLR ranks 13 and 103. Save the
selected ordinals and elementwise reference outputs locally for the after-run;
record reproducible source, timing scope and timing/error CSVs in the repo.
Separate productive contributions from pruning-only calls. This complements
the earlier numerical-segment microbenchmarks and remains CPU-only.

Implementation remains staged: output GEMM on padded storage, then packed
active views, then capacity-aware convolution scratch and Action integration.
No layout and convolution changes will be hidden in the structural commits.

## Reusable capacity and padded GEMM outputs — 2026-11-03

Source checkpoints: `df95898` `[cppdlr] reuse convolution capacity` and
`6565624` `[bs] reuse vertex outputs`.

The scalar workspace retains its existing four-dimension constructor. Each of
its five arrays now reserves capacity; calls construct packed active views
whose rank and required element count fit that owner. The numerical formula
and batched overload are unchanged. Allocation guards reject integer/byte
overflow before resizing. Tests cover consecutive rectangular growth/shrink,
stable pointers, untouched tails, shape-changing output aliases, insufficient
individual buffers, and boundary-layer kernels after a larger call.

Sparse vertex GEMMs now write into the spare edge buffer, apply the vertex
kernel, then swap ownership with T or U. Later views are recreated, and a live
forward T is never swapped during backward traversal. Endpoint products are
all saved in Tkaps before T is cleared; endpoint accumulation uses GEMM with
beta one. Self-energy scaling touches only its active result. Padded views
and fixed cppdlr convolution are deliberately retained at this checkpoint.

The focused scalar/batched convolution and sparse prepared/path tests all
passed (4/4 CTest entries, 0.46 s). Independent review found no capacity,
aliasing or lifetime blocker. The matched pre-change benchmark captured 56
productive/zero-completion cases and a 4 MiB local elementwise reference;
its source and compact results will be committed separately.

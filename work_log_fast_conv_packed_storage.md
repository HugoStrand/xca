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

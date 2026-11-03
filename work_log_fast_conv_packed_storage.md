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

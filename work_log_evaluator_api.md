# Evaluator API work log

## 2026-10-13 — Scope and implementation checkpoint

User requested a unified `_triqs_xca_cuda` wrapper, GPU helper extraction to `gpu.py`,
and CPU range bounds instead of MPI index vectors. Work is on `dev_cuda`; preserve
unrelated changes and all experimental sources/results. Code and this log use separate
commits with terse bracketed messages.

Chosen interface: half-open `[first,last)` range methods matching native-small.
The cuBLAS evaluator also receives range methods so Solver uses one work partition
for every backend. Paired ranges cover the ordinary index domain and internally
filter forward seeds. Negative, inverted and out-of-domain ranges fail before traversal;
empty tails return correctly shaped zero outputs. Explicit noncontiguous C++ index
selections remain for preserved experiments/benchmarks, excluded from CPU Python wrapping.

CUDA class aliases and runtime functions now share one extension. Optional small-kernel
registration is guarded after clair generation, keeping library-only CUDA builds valid.
Device selection, lazy imports and collective error coordination live in `gpu.py`.
Verification and implementation commits follow as work completes.

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

## 2026-10-13 — Verification and completion

Implementation commit: `482146c` (`[solv] evaluator ranges and CUDA bindings`).

- Consolidated both evaluator classes and runtime functions in `_triqs_xca_cuda`.
  Removed all `_dense_cuda.*` and `_dense_cuda_small.*` source wrappers.
  Regenerated CPU and combined CUDA bindings with clair; the generator postprocessor
  guards optional native-small declarations and registration. CPU imports remain lazy.
- Extracted backend loading, process device selection and collective error handling
  to `gpu.py`; removed the old helpers and index-array partitioner from Solver.
- CPU dense/block-sparse and cuBLAS evaluators expose checked half-open range methods
  for their existing observables. Solver uses bounds for all backends. Paired ranges
  filter forward seeds internally. Noncontiguous C++ selection overloads are retained
  for benchmarks and experiments and are excluded from CPU Python wrapping.
- Added C++ tests for full versus partitioned ranges, empty tails, invalid bounds,
  recovery, paired boundary filtering, correlator equivalence and long-index ranges.
  Python tests cover actual dense/block-sparse MPI reductions, module consolidation,
  CPU vector-selector rejection, device selection, dynamic channels and serialization.
- Updated the block-sparse plan's API paragraph to reflect completed range migration.

Verification:

- CPU configure/build passed; complete CTest suite 68/68 passed in 49.46 s.
- CUDA configure/build passed; complete CTest suite 76/76 passed in 114.89 s.
- After final formatting and additional Python checks, both builds passed again.
  Focused C++ range/long-index tests passed in both trees. Final Python integration
  tests passed: 14 cases (CPU skips 7 unavailable CUDA cases; full CUDA skips none).
- Two-rank MPI Python integration runs passed for CPU and full CUDA, 14 cases per
  rank with the same capability-based skips. This includes the CPU dense/block-sparse
  range reductions and coordinated GPU initialization/evaluation errors.
- An isolated `cbuild-cuda-library-api` build with `XCA_CUDA_SMALL=OFF` compiled the
  combined extension and supporting modules, using symlinks to existing dependency
  source caches. Its 14 Python cases passed (3 native-small skips); module layout
  confirms the cuBLAS class is exported, native-small is absent, and CPU package import
  does not load the CUDA extension.
- `git diff --check` passed. Existing benchmark/experiment sources and data are unchanged.
  Unrelated `.gitignore`, AGENTS files and untracked tests are preserved. No push or
  install-prefix changes were made. Main build trees remain configured for CPU/full
  CUDA, with binding regeneration OFF after targeted regeneration for verification.

Temporary logs use `/tmp/xca-api-{cpu,cuda,library}-*`; they may disappear. The recipes
follow AGENTS environment switchers and numerical-library thread settings. The initial
extra block-sparse Python fixture used None instead of Solver's 'automatic' sentinel;
corrected the fixture and reran the serial/MPI checks successfully.

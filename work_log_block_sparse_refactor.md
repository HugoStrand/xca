# Block-sparse evaluator refactoring work log

## Reviewed plan — 2026-11-03

Plan: `plan_block_sparse_refactor.md`, on `dev_backbone`.

The sparse evaluator already has typed backbones/paths, a const Action, a
private self-energy accumulator and rank-local topology evaluation. The
remaining dense patterns map to two coherent stages: clarify/hide the public
state and return native owning numerical results, then extract cpp-local
numerical helpers with explicit inputs and accumulators.

Review preserves sparse-specific path discovery, segment reuse, support tables,
endpoint selection and sector-varying output. In particular, prepared self-
energy returns `BlockDiagOpFun` rather than forcing varying blocks into a dense
array. Topology methods retain their Python-facing block-GF output and existing
MPI partitioning. Prepared correlator names mirror dense `compute_*`; internal
accumulating functions keep `eval_*`.

Tests currently inspect public scratch during malformed-path checks; replace
that internal coupling with returned-result ownership, recovery and call-
interleaving checks. Generated wrapper properties are unused by repository
Python callers and will disappear when state becomes private. Fixed cppdlr,
padded storage and numerical traversal stay unchanged in this structural pass.

After the full CPU validation and review checkpoint, complete this persistent
goal and start the explicitly requested goal for
`plan_fast_conv_packed_storage.md`. Code and documentation commits remain
separate; dense fast-convolution/batching work is concurrent but independently
reviewable.

## Public API and private state — 2026-11-03

Source: `[bs] clarify evaluator api`.

Made sparse scratch/reset private, removed duplicate scalar members, and
renamed prepared correlator calls to `compute_one_time_correlator`. Prepared
self-energy now returns independent native `BlockDiagOpFun` storage; topology
methods retain block GFs and the existing Python signatures. Native inputs
are const. Tests and the backbone benchmark now use Action/model metadata.

Migrated scratch-inspecting path tests to retained-output ownership, changed-
propagator interleaving, rejected-input recovery and empty-selection checks.
Compile-time checks enforce typed/native result signatures and hidden state.
The first stage keeps the current internal numerical methods and arithmetic.

Regenerated `_triqs_xca.wrap.cxx` with the installed newer clair. Besides the
removed unused sparse member properties, generated identifiers, type spelling
and generator-version metadata changed mechanically. Constructors and topology
methods retain their semantics. Python and MPI API tests passed.

Full CPU build passed. The full suite passed 84/85 entries in 33.86 s, including
every sparse/path/layout/API check and both serial/three-rank MPI suites. The
single failure compares sixth-order dense-fast and sparse-fixed convolution
at 1e-13 relative tolerance; observed discrepancy is 2.3e-13 (4.8e-21 absolute).
An isolated pre-integration dense reference comparison is investigating this
concurrent numerical change before final acceptance. No sparse tolerance was
changed. Logs: `/tmp/xca-sparse-api-full-{build,test}.log`.

## Explicit numerical helpers — 2026-11-03

Source: `[bs] extract evaluator kernels`.

The class now contains owned state, private reset, constructors and public
operations. Anonymous-namespace helpers receive explicit Action, propagator,
BackbonePath, scratch and output dependencies. A single adjoint-selected
single-particle operator helper replaces two copies; general observable matrix
conversion is also file-local. Prepared self-energy accumulates directly in
its returned native container, removing the intermediate member-result copy.

Sparse-specific discovery, support tables, endpoint checks, cached segments,
reflection and numerical grouping remain. Correlator helpers preserve each
backbone's sum/sign before adding it to the caller's output. This introduces
one extra output-sized local accumulator for prepared correlator calls; the
topology allocation pattern is unchanged. It is explicitly deferred to later
workspace optimization. No padded layout, vertex matmul or convolution was
changed in this refactoring checkpoint.

Two independent reviews checked view lifetimes, T/U separation, native-result
ownership, exact-zero blocks and early validation; no blockers remained. The
focused path, prepared and high-index regression entries passed (3/3, 0.19 s),
and both existing/new benchmark targets built. Full CPU acceptance follows
the coordinated rebuild. The high-index discrepancy was independently traced
to dense convolution arithmetic and addressed only in the cross-engine test,
as recorded in `work_log_dense_cpu_batched.md`.

## Accepted structural checkpoint — 2026-11-03

The final full CPU build passed and all **85/85 CTest entries passed** in
33.46 s, including Python wrappers and actual three-rank C++/Python MPI
evaluation. Logs: `/tmp/xca-sparse-refactor-final-{build,test}.log`.
The two source checkpoints are `1ddbb7b` and `78576a6`; no packing or sparse
convolution change is included. This completes the structural refactoring
goal and permits the separate packed-storage/fast-convolution goal to start.

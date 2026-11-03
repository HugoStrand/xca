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

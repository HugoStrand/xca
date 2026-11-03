# Mirror the dense evaluator cleanup in block-sparse evaluation

Completed on 2026-11-03 in source commits `1ddbb7b` and `78576a6`, with 85/85
CPU CTest entries passing before the separate numerical/storage work began.
See `work_log_block_sparse_refactor.md` for the step-by-step validation record.

## Purpose and boundaries

Make `block_sparse::DiagramEvaluator` as readable as the dense evaluator: its public methods describe requested results, its private members own reusable numerical storage, and file-local functions implement the arithmetic with explicit inputs and outputs. Retain the sparse-specific path discovery, support tables, endpoint sums and correlator segment reuse. This is a structural pass before implementing `plan_fast_conv_packed_storage.md`.

The current dense public names are `compute_self_energy`, `compute_self_energy_by_pairs`, `compute_single_ptcle_gf` and `compute_one_time_correlator`; mirror the applicable `compute_*` names rather than introducing a second `evaluate_*` vocabulary. Internal accumulating functions use `eval_*`. Do not add an abstract evaluator hierarchy or force sparse evaluation into dense storage.

No packed storage, GEMM rewrite, fast convolution, traversal reordering, new numerical cutoff or changed MPI partition/reduction belongs in this pass. Preserve Python constructors and topology-call signatures. Fast convolution is being adopted independently in the dense evaluator, initially owned by that evaluator.

## Review findings

The sparse evaluator already has typed self-energy/correlator backbones and paths, a const owned Action, a private Sigma accumulator, and topology calls that return each MPI rank's local sum. These pieces need no redesign.

Remaining differences from dense:

- Scratch arrays `T`, `U`, `GKt`, `Tkaps`, `Tmu` and `reset()` are public. Generated Python bindings expose writable scratch-array properties. No repository Python caller needs these properties or the scalar properties.
- `beta`, `r`, `n`, `n_hyb`, `n_int` and `q` duplicate Action information; `Nmax` duplicates the maximum block dimension used to size scratch. Tests and the existing backbone benchmark inspect some of these fields, so their uses must migrate to Action/model information rather than retaining redundant accessors.
- Prepared correlator calls are named `eval_correlator`, whereas dense uses `compute_one_time_correlator`. Prepared sparse self-energy calls return TRIQS block GFs, mixing numerical evaluation and presentation. The natural numerical result is the existing owning `BlockDiagOpFun`, not one rectangular ndarray because sector sizes differ.
- About twenty private numerical/lookup methods borrow member data implicitly. Vertex multiplication, edge convolution, endpoint contraction, segment traversal and observable completion can be anonymous-namespace functions taking Action, propagator, path, scratch and output explicitly.
- Separate single-particle annihilation/creation setup methods duplicate almost all their code. One `single_particle_operators(bool adjoint, Action const&)` helper can select either operator family, as in dense. The general observable conversion helper should also have file-local linkage.
- Prepared calls accept mutable propagators although path discovery and numerical evaluation only read them. Use `BlockDiagOpFun const&` throughout these boundaries.
- `self_energy_result` converts to block GFs and resets evaluator state as a side effect. Separate conversion from reset. Prepared calls should create independent numerical results; topology calls accumulate directly before a single presentation conversion.

## Commit 1: clarify the public API and owned state

Suggested code commit: `[bs] clarify evaluator api`

1. Keep public const Action and constructors. Keep Python-visible topology methods unchanged.
2. Make workspace and reset private; remove redundant scalar members. Use short local `r`, `beta`, or maximum-dimension variables where repeated use warrants them, derived from Action or workspace shape. Keep no scalar merely to preserve a Python property that has no caller.
3. Rename the two prepared correlator overloads to `compute_one_time_correlator`; update C++ tests, test fixture adapters and the backbone benchmark. These remain `C2PY_IGNORE`.
4. Prepared self-energy overloads for a Backbone or a complete BackbonePath return an owning `BlockDiagOpFun`. Topology input continues to return `block_gf<dlr_imtime>`. Test adapters provide the existing presentation conversion where shared tests need it. This explicitly mirrors dense's numerical-array versus Python-facing block-GF boundary without inventing a new sparse container.
5. Accept propagators by const reference at numerical boundaries. Keep typed path constraints and the saved-path validity contract, including rediscovery when activity masks change.
6. Regenerate bindings; inspect the diff to ensure only obsolete sparse member properties disappear. Preserve all four Python constructor variants and the three applicable topology methods.

Tests affected include `block_sparse/dynint_evaluator`, `dynint_spin_flip`, `prepared_evaluator`, `path_evaluator`, `propagator_layout`, `test_utils/prepared_evaluators.hpp`, and `benchmarks/backbone_cpu`. General comparison fixtures must continue returning their existing common representation.

## Commit 2: make numerical dependencies explicit

Suggested code commit: `[bs] extract evaluator kernels`

1. Move direct Action lookup, propagator validation and conversion, vertex/edge kernels, endpoint contraction, segment traversal and correlator completion into the cpp anonymous namespace. Use explicit ndarray/view types and ordinary return-type syntax. Avoid passing the entire evaluator as a substitute for explicit dependencies.
2. Implement accumulating `eval_self_energy` / `eval_correlator` helpers taking their output by reference. The self-energy helper for a prepared backbone discovers paths and sends them to the same path kernel as the complete-path public entry point. Correlator backbone evaluation retains segment caching; complete-path evaluation remains independently usable.
3. Preserve traversal and accumulation order, exact-zero support behavior, nonclosing self-energy rejection, endpoint capacity checks, reflection and all signs. The existing padded storage and allocating convolution remain unchanged so numerical/layout work is reviewed separately.
4. Keep the evaluator responsible for allocating scratch, selecting the public operation and MPI share, initializing results, and returning owned results. No result view may escape with lifetime tied to scratch.
5. Merge the two single-particle operator helpers with an adjoint selector. Keep generic observable construction separate and file-local. Do not change the occupation-basis conversion or observable statistics checks.
6. If explicit argument lists become lengthy, keep a few focused helper layers; do not introduce a new workspace hierarchy, execution context or virtual dispatch merely to shorten signatures.

## Validation and review checkpoints

Use existing coverage first: physical dense/sparse comparisons, dynamical interactions, varying sector dimensions, prepared-backbone/path equivalence, tiny nonzero coefficients, malformed paths, mismatched propagator layouts, and MPI topology sums. Rebuild and run the focused sparse/diagram tests after each coherent stage; regenerate and test Python wrapping after the public-state stage.

Add narrowly targeted API/lifetime coverage where missing: const propagator calls; prepared self-energy's `BlockDiagOpFun` return type; absence of public reset/scratch/scalars; retained result ownership after later calls; repeated self-energy/correlator interleaving; recovery after rejected inputs. Existing path tests already cover empty endpoint selections and complete-path requirements, so extend them rather than duplicate their fixtures.

Run the full CPU CTest suite after both stages and build the affected benchmark target. Do not benchmark this structural pass as an optimization claim. Review generated bindings and inspect git diff for unrelated edits. Code and step-by-step work-log updates are committed separately with terse house-style messages.

## Handoff to packed storage and fast convolution

Once this refactor is validated, complete its persistent goal and create the explicitly requested goal to implement `plan_fast_conv_packed_storage.md`. The now-explicit kernel scratch/output dependencies should make active-view packing and output GEMM reuse changes local. Preserve the existing benchmark source/results as the baseline. Move immutable fast-convolution tables into Action only when sparse integration is ready; mutable convolution workspace remains evaluator-owned.

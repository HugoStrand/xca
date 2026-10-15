# Backbone refactoring work log

## Review index

| Stage | Code commit | Status |
| --- | --- | --- |
| Planning | None | Review process and staged draft recorded |
| Test dependency review | None | Backbone coupling inventoried in the plan |
| Plan review | None | Revised implementation plan ready for execution |
| 0 | None | CPU configure/build and all 66 CTest entries passed |
| 1 | `0d70b28` | Self-energy path discovery extracted; all 67 CTest entries passed |
| 2a | `7695377` | Owned assignment data consumed by path discovery; all 67 CTest entries passed |
| 2b | `4f05c6e` | Shared preparation owns diagram rules; all 67 CTest entries passed |
| 3a | `e932ea1` | Explicit statistics and prepared permutation signs; all 67 CTest entries passed |
| 3b–7 | None | Numerical consumer migration and remaining stages pending |

## Planning checkpoint — 2026-10-15

The working branch is already `dev_backbone`, based on `origin/dev_hs` at
`2b0fc5c`. No checkout or branch creation was performed for this checkpoint.
The development sequence is drafted in [plan_backbone.md](plan_backbone.md).

The main requirement is that a reviewer can follow responsibility transfers in
the existing CPU evaluators, one meaningful code commit and corresponding log
commit at a time. Working entry points remain available during extraction;
temporary adapters are removed at the final consolidation stage. Numerical-policy
changes and optimization are reviewed separately from structural extraction.

Read order: the plan's review requirements, migration approach, then proposed
stages. The initial concrete seam is self-energy path discovery; subsequent
stages separate shared diagram preparation from block-specific planning and
numerical execution. The sequence is provisional until the detailed plan is
refined against the implementation.

Validation for this checkpoint was inspection of Git state, development
instructions, backbone declarations, shared evaluator test infrastructure, and
the CPU path/cutoff sites. No code was changed and no builds or benchmarks were
run. Existing unrelated worktree changes were left untouched. Next: complete the
detailed implementation plan before beginning stage 0.

## C++ test dependency review — 2026-10-15

Inspected the C++ test tree for backbone construction, mutation, internal
evaluation calls, and indirect enumeration assumptions. Five translation units
construct backbone objects: the indexing and parity tests, block-sparse sparsity
invariance tests, dense correlator-statistics tests, and the shared one-time
correlator tests. Flat-index tests and the shared evaluator adapter add coupling
to enumeration and paired selection; one error assertion names `set_flat_index`.

The plan now maps these dependencies to migration stages. Numerical invariants,
hand-derived expectations, and negative controls must survive changes to the
test harness. Legacy reset/sentinel/error-message details may change explicitly
as their implementation disappears. Shared evaluator agreement is supplemented
by analytic tests because the reference evaluator and some test helpers share
production preparation logic.

Read the plan's "Existing C++ test dependencies" table before the staged
sequence. No source or test code was modified and no tests were run: this was a
static dependency review and documentation update. Implementation remains pending.

## Implementation plan review — 2026-10-15

Reviewed the draft against the actual self-energy walker and the existing test
contracts. The major risk was hiding semantic changes inside extraction: the
walker does not explicitly enforce final-sector closure, and its arithmetic
still mutates special endpoint indices. The revised stage 1 records terminal
sectors, preserves current filtering, and names the temporary mutable dependency.
Owned metadata is streamed with bounded memory rather than collected globally.

Stages 2–3 now describe smaller review units for preparation and consumer
migration. Stage 6 distinguishes numerical zero policy from support/grouping
semantics, with explicit treatment of cross-set remainders. It does not require
an optimal grouping algorithm or a new sparse numerical backend. Completion
requires all observable paths, numerical-policy validation, adapter removal, and
full CPU tests. CPU execution remains the scope; CUDA migration comes later.

Next: establish the configured CPU baseline and semantic checklist, then extract
self-energy path discovery with its immediate caller and focused path tests.

## Stage 0 — semantic checklist

The initial review example is a first-order diagram with one fermion and the
empty/occupied sectors: either direction leaves one allowed sector path and
returns to its starting sector. The shared `unequal_sets` and `unequal_sets_nA0`
fixtures extend the checks to unequal groups and a density interaction, with
partitioned and dense versions of the same model. Existing analytic one- and
two-fermion tests remain independent checks on whole diagram results.

| Responsibility | Current owner and migration constraint |
| --- | --- |
| Line direction and endpoint conjugation | `Backbone::set_directions`; dagger choices must agree with the barred-operator convention. |
| Stable scalar kernels | `Backbone::set_pole_inds` chooses edge/vertex kernel placement and inverse powers at zero; `hyb::Hybridization` applies the kernels numerically. Pole signs, reflection, and prefactors must move together. |
| Fermionic permutation | `Backbone::get_parity`, called after setting special endpoint orbital indices; correlators use external-statistics sentinels. The explicit-statistics replacement must preserve mixed interaction signs. |
| Special endpoint sum | Dense and block-sparse `multiply_left_vertex_and_right_zero_vertex` sum endpoint indices and transpose coefficient indices on the backward branch. Dense hybridization reflection includes a minus sign; block-sparse code applies that sign explicitly. |
| Raw self-energy sign | C++ applies the order sign, first-line direction sign, backbone prefactor, and permutation parity. Python subsequently subtracts raw topology results. The `sign` yielded by Python topology enumeration is not multiplied again. |
| Observable sign | C++ computes raw correlators with external statistics, while Python applies `(-1)^order` for single-particle and general one-time correlators. |
| Partition of work | Python distributes flat-index vectors over MPI. Dense paired self-energy selects seeds with first-line direction zero using the enumeration stride, then C++ evaluates both directions. Preserve complete coverage without double counting. |
| Input/output masks | `BlockDiagOpFun` construction and `set_zero_block_indices` classify blocks using `1e-16`; masks affect traversal and accumulation. The evaluator also drops small `T_out` contributions. These policies remain temporarily during structural extraction. |
| Symmetry restrictions | `check_sym_set_block_diagonal` permits cross-set coefficients up to a relative `1e-12` threshold; path traversal nevertheless admits only equal special-endpoint sets. Stage 6 must account for that remainder explicitly. |

The first extraction will preserve the traversal and summation order. Discovery
will report output/terminal sectors and dimensions without invoking numerical
evaluation. It will still use the existing quartet connectivity and propagator
mask; it does not yet establish exact per-operator support or remove numerical
cutoffs. These limitations are deliberate and reviewed in later stages.

Baseline validation: sourced `~/env/xca-cpu.sh`, ran `bash ../cmake.sh` from
`cbuild`, then `cmake --build cbuild --parallel 6` and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild --parallel 2
--output-on-failure`. Configure/build succeeded; all 66 entries passed in 58.43 s.
Binding regeneration produced no tracked changes. Logs are temporarily available
as `/tmp/xca-backbone-baseline-{configure,build,test}.log`. The untracked CUDA smoke
test remains present and registers its disabled CPU case; this is not GPU
validation. The baseline includes the existing local test files discovered by
CMake, and unrelated worktree changes were preserved.

## Stage 1 — self-energy paths (`0d70b28`)

Previously `DiagramEvaluator::find_path_self_energy` decoded a flat index,
searched sector paths, and immediately performed each contraction. Connectivity
could not be inspected without entering the evaluator. Now
`visit_self_energy_paths` discovers metadata and calls a consumer with an owned
`SelfEnergyPath`. The evaluator's `eval_self_energy_paths` is the small bridge:
decode the assignment, visit paths, call the existing arithmetic, reset the
backbone. This is the first responsibility transfer, not a replacement evaluator.

Read [SelfEnergyPath](c++/triqs_xca/block_sparse/self_energy_path.hpp), then
[visit_self_energy_paths](c++/triqs_xca/block_sparse/self_energy_path.cpp), then
`eval_self_energy_paths` in
[diagram_evaluator.cpp](c++/triqs_xca/block_sparse/diagram_evaluator.cpp).
The arrays store successive propagator sectors and vertex dimensions, with
separate output/terminal sectors and special endpoint symmetry sets. One path
workspace is reused; copies retain their own arrays. The walker reads operator
sets through const references and performs no diagram contractions.

The four [path tests](test/c++/block_sparse/self_energy_path.cpp) use explicit
synthetic connectivity with sector dimensions 1,2,1. For example, two creation
vertices followed by two annihilation vertices traverse 0→1→2→1→0, giving
propagator sectors [1,2,1] and dimensions [1,2,1,2,1]. They check both first-order
directions, symmetry-set ordering, retained copies, masked propagators, an empty
output block, crossing/nested paths, and workspace dimensions. A deliberately
non-adjoint connectivity fixture verifies that terminal sectors are reported
without adding a new closure filter; it is not a valid physical model.

No existing test assertion or public evaluator API changed. Numerical operations,
traversal order, equal-set restrictions, and cutoff decisions are retained.
The walker still reads the legacy backbone and quartet, so this descriptor is
not yet a self-contained numerical task. The evaluator callback still mutates
special endpoint indices, which are not walker traversal inputs; stages 2–4
remove that coupling. Broader support/closure policy remains stage 6 work.

Validation: CPU configure/build succeeded with binding regeneration unchanged.
All 67 CTest entries passed in 56.53 s with the baseline thread settings. All four
focused GoogleTests also passed after adding the crossing-path case. Logs:
`/tmp/xca-backbone-path-{configure,build,test}.log` and
`/tmp/xca-backbone-path-focused-{build,test}.log`. Next: stage 2, owned shared
assignment/topology data, consumed first by the path walker to remove its
dependency on mutable backbone state.

## Stage 2a — owned diagram assignments (`7695377`)

Path discovery previously read the same mutable backbone as numerical execution.
It now consumes a const `DiagramAssignment`: topology, vertex selections, arc
directions, internal pole indices, and the scalar-kernel recipe are owned data.
`DiagramTopology` carries only arcs. Neither type contains block sectors or
matrix/workspace storage. `snapshot_assignment` is the temporary bridge from the
legacy object, called immediately before traversal. Resetting or destroying that
source no longer invalidates the prepared description.

Read [diagram_assignment.hpp](c++/triqs_xca/diagram_assignment.hpp), its
[snapshot implementation](c++/triqs_xca/diagram_assignment.cpp), then the changed
selector inputs in
[self_energy_path.cpp](c++/triqs_xca/block_sparse/self_energy_path.cpp).
The evaluator still performs its existing arithmetic through the legacy backbone.
Special endpoint orbital indices are deliberately unresolved (`-1`) in the
assignment, even after an earlier contraction has mutated them: they belong to
the remaining endpoint sum or observable, not the internal assignment.

The hand-derived expectations in
[backbone_indexing.cpp](test/c++/backbone_indexing.cpp) now also check the snapshot's
vertices, directions, poles, edge kernels, and prefactors. No existing assertion
was removed. A correlator case verifies the absent external arc, unresolved
endpoints, source destruction, and deep copying of every array. The path tests
were migrated to the new input type and now reset/reassign the old backbone
inside callbacks to verify that subsequent discovery is independent of it.

Validation: CPU configure/build and the focused indexing/path CTest entries
passed; all 67 CPU CTest entries passed in 56.80 s. Binding regeneration left
tracked wrappers unchanged. Logs are
`/tmp/xca-backbone-assignment-{configure,build,focused,test}.log`.

This step adds a metadata copy per assignment, bounded by diagram order, and
makes no performance claim. The snapshot bridge is temporary: stage 2b extracts
decoding and stable-kernel preparation into this shared layer and makes the
legacy API delegate to those rules. The existing mutable setters and the
numerical consumers have not yet been replaced. The goal remains active.

## Stage 2b — shared diagram preparation (`4f05c6e`)

The snapshot bridge has been removed. Preparation now starts with a validated
`DiagramTopology` and logical `AssignmentIndices` (directions, poles, and internal
orbitals), or uses a separate flat-index decoder to obtain those selections.
`prepare_vertices` and `prepare_kernel_factors` own the direction/bar/dagger and
stable scalar-kernel rules previously embedded in the mutable backbone setters.
Both direct preparation and the legacy setters call those functions. The
block-sparse path walker consumes the resulting description directly.

Read the public functions in
[diagram_assignment.hpp](c++/triqs_xca/diagram_assignment.hpp), then
[diagram_assignment.cpp](c++/triqs_xca/diagram_assignment.cpp), then
`Backbone::prepare_assignment` in [backbone.cpp](c++/triqs_xca/backbone.cpp).
The last function is a temporary arithmetic adapter: prepare the description,
load the old fields, and return the description. Numerical consumers still use
the old fields until stages 3–5 migrate them. It does not implement a second set
of diagram rules. The legacy component-index setters remain compatibility APIs;
their eventual removal is still stage 7 work.

Logical preparation has no requirement that all assignments fit in a `long`.
A new test prepares one order-18, 14-flavour assignment even though counting
its full flat enumeration overflows. Flat selections retain their existing
orbital/pole/direction digit order, bounds checks, and overflow protection.
Logical descriptions use `flat_index = -1` to indicate that no enumeration
identity was assigned. First-order and correlator cases retain their respective
direction counts; special endpoint orbitals remain unresolved.

One intentional behavioral improvement follows from complete value preparation:
calling `set_flat_index` again no longer needs a preceding reset to avoid stale
edge kernels or a stale prefactor sign. A regression checks vertex-to-edge
placement on reassignment and preservation after invalid input. Existing
evaluators already reset between assignments, so their diagram sums are
unchanged. Numerical cutoff and support policies have not changed.

Test migration: the hand-derived indexing expectations now exercise direct
logical preparation, not a snapshot of legacy state. Path tests prepare their
inputs directly; the lifetime case retains the arithmetic adapter explicitly.
New checks cover mixed-radix decoding, zero poles in both orientations, invalid
logical indices, and unrepresentable enumeration counts. The assertion requiring
`set_flat_index` in exception text was deliberately replaced by a relevant
`flat index` diagnostic check; rejection and call-recovery tests remain intact.

Validation: CPU build and all four focused CTest entries passed. All 67 CTest
entries passed in 57.07 s. After adding the enumeration-overflow independence
case, all eight indexing/preparation GoogleTests passed. Binding regeneration
left tracked wrappers unchanged. Logs:
`/tmp/xca-backbone-preparation-{build,focused-build,focused,indexing,test}.log`.
Next: explicit interaction/observable statistics and sign preparation, followed
by dense numerical consumer migration. Stages 3–7 remain required.

## Stage 3a — prepared permutation signs (`e932ea1`)

Both evaluators previously changed special endpoint orbital indices inside the
matrix evaluation to ask `Backbone::get_parity` for a sign. They now prepare two
signs before the contractions: one with the special arc in the fermionic
topology, and one without it. The endpoint loop selects the first only when both
endpoint operators are fermionic. This captures all endpoint statistics without
expanding the orbital-pair sum or recalculating parity for every pair/path.

Read [diagram_sign.hpp](c++/triqs_xca/diagram_sign.hpp), its
[implementation](c++/triqs_xca/diagram_sign.cpp), then the special endpoint
contractions in the [dense](c++/triqs_xca/dense/diagram_evaluator.cpp) and
[block-sparse](c++/triqs_xca/block_sparse/diagram_evaluator.cpp) evaluators.
`Statistics` is explicit metadata; preparation accepts arbitrary flavour order.
The current model boundary still constructs that metadata from its existing
hybridization-then-interaction layout. General interaction/operand preparation
remains subsequent work; this is not yet a full interaction catalogue.

Neither evaluator now calls `get_parity` or writes the special endpoint indices.
Correlators select their prepared sign from explicit external statistics rather
than `-2`/`-3` sentinels. The old `get_parity` method remains a compatibility
translation for direct callers/tests and delegates the sign calculation to the
same shared function. Order, first-line direction, scalar-kernel, reflection,
and Python observable prefactors are unchanged and remain separate factors.

The hand-derived table in
[backbone_parity.cpp](test/c++/backbone_parity.cpp) now checks explicit preparation
directly and also verifies the legacy translation. Sentinel/default-constructor
checks remain while that adapter exists. Additional cases verify interleaved
statistics, mixed endpoint statistics, correlator external statistics, invalid
internal selections, and first order with no internal flavours. The existing
forced-statistics negative controls in the dense correlator tests were retained.

Validation: CPU configure/build, five sign-focused CTest entries, and all 67
CPU CTest entries passed (full suite 56.71 s). No tracked binding changes.
Logs: `/tmp/xca-backbone-signs-{configure,build,focused,test}.log`. No benchmarking
or numerical-policy change was performed. Next: migrate dense operator/kernel
consumption to the prepared assignment; the remaining mutable numerical reads,
block-path orchestration, support policy, and adapter cleanup are still required.

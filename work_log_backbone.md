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
| 3b | `3832492` | Dense arithmetic consumes const assignments; all 68 CTest entries passed |
| 3c | `8ff3717` | Block-sparse arithmetic consumes const assignments; all 69 CTest entries passed |
| 3d | `6de60bc` | Shared directed interactions and partial-contraction descriptions; all 70 CTest entries passed |
| Materialization refresh | `5234d69` | Dense coefficient updates replace old bars; all 70 CTest entries passed |
| 4a | `984e4bc` | Self-energy operands bound before arithmetic; all 70 CTest entries passed |
| 5a | `eff9128` | Correlator fragment and join planning extracted; all 71 CTest entries passed |
| Shared kernel binding | `18d5cd5` | Scalar factors bound once per assignment; all 71 CTest entries passed |
| Remaining 3–7 | None | Shared model/observable metadata, numerical/support policy, and interface cleanup pending |

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

## Stage 3b — dense prepared evaluation (`3832492`)

Dense matrix/convolution routines now read `DiagramAssignment const &` rather
than mutable backbone getters. The prepared entry points accept logical
assignments without a flat index or backbone object. Existing entry points
remain selection adapters: they prepare an assignment from topology/index and
call the numerical routine. Dense evaluation no longer sets, resets, or reverses
a backbone, including in its correlator and paired paths.

Read the prepared `eval_self_energy`, `eval_self_energy_pair`, and
`eval_correlator` overloads in
[dense/diagram_evaluator.cpp](c++/triqs_xca/dense/diagram_evaluator.cpp), then
the vertex binding in [dense/fset.cpp](c++/triqs_xca/dense/fset.cpp) and the
assignment-consuming kernel overloads in [hyb.cpp](c++/triqs_xca/hyb.cpp).
An internal vertex selects its dense operator from the prepared bar/dagger,
orbital, and pole data. Special endpoint operators are selected by an explicit
orientation. The paired routine applies both orientations to its shared
intermediate without mutating the assignment. Kernel multiplication order,
convolution orientation, reflected correlator kernels, and prefactors are
unchanged.

The new [prepared evaluator tests](test/c++/dense/prepared_evaluator.cpp) use a
model with a density interaction. They compare logical execution with indexed
selection, a paired contribution with two ordinary contributions, and retained
assignments with changed propagators. Scaling G by 1.1 scales the order-two
self-energy by 1.1^3 and the correlator by 1.1^4, checking that preparation does
not retain numerical propagator data. These are interface/lifetime checks;
independent diagram accuracy still comes from the existing analytic suites.
Prepared calls accumulate self-energy into Sigma or return a correlator, as
documented in the header. Their indices must refer to this evaluator's model and
pole tables. Existing direct-backbone tests continue through the adapters.

Validation: CPU configure/build, six focused CTest entries, and all 68 CPU CTest
entries passed (full suite 56.92 s). No tracked binding changes. Logs:
`/tmp/xca-backbone-dense-{configure,build,focused,test}.log`. Source inspection
also confirms no dense numerical reads of backbone vertex/pole/kernel state.

This is the dense consumer migration, not completion of stage 3. Matrix operand
binding still interprets the prepared vertex forms; the common description of
interaction contractions and block-sparse operand binding remain to be made
explicit. Legacy hybridization overloads remain for block-sparse arithmetic and
must be removed after that migration. Stages 4–7, including support/numerical
policy and final adapter cleanup, remain required.


## Stage 3c — block-sparse prepared evaluation (`8ff3717`)

Block-sparse numerical routines now consume the same const `DiagramAssignment`
as self-energy path discovery. Previously discovery read the prepared value but
the arithmetic still read the mutable backbone used to create it. Vertex, edge,
prefactor, special-endpoint, and correlator routines now use the value directly.
No evaluator numerical routine needs backbone selection, reset, or reversal.
The remaining backbone use is in selection/count adapters and diagnostic printing.

Read the prepared `compute_self_energy` and `eval_correlator` overloads in
[block_sparse/diagram_evaluator.cpp](c++/triqs_xca/block_sparse/diagram_evaluator.cpp),
then `eval_self_energy_paths` and the vertex/edge helpers. Indexed calls prepare
an assignment and delegate; whole-diagram loops reuse their path workspace.
The prepared self-energy entry returns one contribution and resets its accumulator
each call. The correlator entry returns its contribution directly. Both use the
current propagator; assignment indices refer to this evaluator's model/pole tables.

With both numerical consumers migrated, the temporary backbone overloads in
[hyb.cpp](c++/triqs_xca/hyb.cpp) were removed, together with that header's backbone
dependency. There is now one implementation of the shared edge/prefactor kernel
operations. Backward coefficient transposition, reflected-kernel conventions,
path traversal, masks, endpoint signs, accumulation order, and cutoffs are unchanged.
This step makes no numerical-policy or grouping change.

The new [block-sparse prepared tests](test/c++/block_sparse/prepared_evaluator.cpp)
compare logical and indexed calls for a density-interaction model and reuse the
same assignment with G scaled by 1.1. The self-energy scales by 1.1^3 and the
identity-operator correlator by 1.1^4 at order two. These tests check selection and
lifetime behavior; the existing analytic and cross-evaluator tests continue to
check the diagram calculation itself. The old direct-backbone tests still work
through the compatibility adapters.

Validation: CPU configure/build, all 14 focused CTest entries, and all 69 CPU
CTest entries passed (full suite 56.93 s). Tracked generated bindings are
unchanged; new prepared C++ methods are excluded from wrapping. Logs:
`/tmp/xca-backbone-bs-assignment-{configure,build,focused,test}.log`.
Source review found no remaining numerical backbone-state reads or callers of
the removed kernel overloads. No CUDA work or benchmarking was performed.

Next: make the shared interaction/partial-contraction and endpoint-sum meanings
explicit, then resolve block-specific operands before arithmetic. Correlator
connectivity discovery still occurs in its evaluation routine; extracting its
fragments and joins remains stage 5. Completing stage 3, numerical/support policy,
selection consolidation, and final adapter cleanup remain required. The persistent
goal is active.


## Stage 3d — interaction contractions (`6de60bc`)

Direction previously had several independent meanings embedded in loops:
operator adjoints, coefficient-index ordering, reflection, and the sign absorbed
into a reflected bar. [interaction.hpp](c++/triqs_xca/interaction.hpp) now owns
these meanings in `InteractionLine`. `BarredOperator` describes a partial sum
with a retained earlier-vertex flavour and fixed pole; each term names the later
operator, coefficient indices, and sign. `EndpointSum` describes the unexpanded
two-flavour sum of the special self-energy arc. None contains matrix storage or
block connectivity, and none requires Cartesian-product materialization.

Read that header first, then the two materialization loops in
[dense/fset.cpp](c++/triqs_xca/dense/fset.cpp) and
[block_sparse/block_op.cpp](c++/triqs_xca/block_sparse/block_op.cpp), followed by
the special endpoint contraction in each evaluator. Vertex preparation also
uses the common direction rules. Dense and block-sparse bars retain their
existing storage and summation order. The block-sparse materializer still limits
its sum to one symmetry set; changing that policy remains a separate stage.

Reversing an interaction transposes coefficient indices; it does not conjugate
coefficients. The operator adjoint is independent of this transpose. The dense
hybridization cache already includes the reflection sign in reflected time
values, while block-sparse evaluation applies it explicitly. These representation
conventions are now documented at the consumers. Permutation statistics and
diagram-order signs remain separate factors.

[interaction_contractions.cpp](test/c++/interaction_contractions.cpp) checks the
directed-line rules and compares both materialized representations with explicitly
written complex, nonsymmetric two-flavour sums. This avoids relying solely on
agreement between two consumers of the same new helper. Existing complex-basis,
retarded-interaction, parity, and correlator tests remain unchanged.

Validation: CPU configure/build, all 11 focused CTest entries, and all 70 CPU
CTest entries passed (full suite 56.60 s). Tracked generated bindings are unchanged.
Logs: `/tmp/xca-backbone-contractions-{configure,build,focused,test}.log`.

The shared mathematical meanings are explicit, but this is not completion of
the planning/execution split. Model-family metadata still enters through the
legacy model adapter, vertex operands are still bound inside arithmetic, and
block correlator paths/joins still need extraction. Stages 4–7 remain required.
Review also identified that `FSet::update_hybridization` does not clear bars when
the pole count is unchanged; verify and correct this refresh/lifetime defect in
a separate code/log pair before relying on model updates.


## Materialization refresh — replace old contractions (`5234d69`)

Review of the new contraction boundary exposed a pre-existing lifetime defect:
`FSet::update_hybridization` only zeroed barred operators when the pole count
changed. A same-size update added the new contraction to the old one; repeated
updates added it repeatedly. This conflicts with the plan's requirement that
numerical materializations not survive incompatible model updates.

Read the refresh test in
[interaction_contractions.cpp](test/c++/interaction_contractions.cpp), then the
three added lines in [dense/fset.cpp](c++/triqs_xca/dense/fset.cpp). After checking
coefficient dimensions and allocating any resized storage, refresh clears both
bar arrays before accumulating the replacement contraction. Invalid dimensions
are still rejected before changing the valid arrays.

The regression was run before the fix and failed, with retained old contributions
producing matrix errors of order one. It now passes for changed same-size complex
coefficients, repeated identical updates, rejected invalid dimensions, and growing
then shrinking the pole count. Expectations are explicit one-operator contractions,
not another call to the refresh implementation. This is a bounded lifetime fix;
it does not introduce evaluator-level model updates or change cutoff policies.

Validation: regression failure recorded in
`/tmp/xca-backbone-refresh-reproduce{,-build}.log`; corrected CPU build and focused
regression passed, followed by all 70 CPU CTest entries (56.94 s). Passing logs:
`/tmp/xca-backbone-refresh-{build,focused,test}.log`. Generated bindings are unchanged.

Next remains explicit model metadata and prepared operands, followed by the
block-specific planning, observable, numerical/support-policy, and selection
cleanup stages. The persistent goal remains active.


## Stage 4a — self-energy operand binding (`984e4bc`)

The block-sparse path walker already supplied sector paths, but matrix routines
still looked up symmetry sets, operator blocks, and endpoint coefficient indices.
The new binder resolves these before execution. This advances the stage-4
boundary using the shared interaction meanings from stage 3d; explicit model-family
metadata and final shared recipe consolidation remain outstanding.

Read [self_energy_contribution.hpp](c++/triqs_xca/block_sparse/self_energy_contribution.hpp)
and its [binder](c++/triqs_xca/block_sparse/self_energy_contribution.cpp), then
`eval_self_energy_paths` and `eval_self_energy_contribution` in
[diagram_evaluator.cpp](c++/triqs_xca/block_sparse/diagram_evaluator.cpp).
A contribution owns its path metadata and borrows views of each internal operator
and propagator. Its special arc contains two endpoint lists, their statistics,
and one coefficient slice per late endpoint. Backward orientation selects the
opposite coefficient axis during binding. No coefficient values are copied and
no Cartesian product of endpoint pairs is stored.

The pipeline is now discover path → bind operands → execute → apply output policy.
The executor reads resolved matrices and the already prepared scalar-kernel
recipe; it performs no symmetry-set lookup, connectivity search, flat-index
decoding, backbone mutation, or output-sector discovery. It returns a view into
its scratch buffer. The caller immediately applies the unchanged `1e-16` cutoff
and accumulates at the bound destination. Keeping this cutoff in the caller makes
the later numerical-policy revision independent of matrix arithmetic.

The contribution must be executed with its matching assignment and mesh, while
the quartet, hybridization, and current propagator remain alive and unchanged.
Rebinding is required after any owner replacement. The current consumer binds
and executes inside one walker callback; it retains no numerical cache across
calls. Metadata copies survive walker workspace reuse. Memory remains linear in
diagram order and endpoint flavours for one contribution. Per-path allocation
cost should be assessed during the final CPU timing checks before optimization.

Left-vertex and edge arithmetic is shared with temporary correlator adapters,
which still perform their own binding. Correlator path discovery and joins remain
stage 5. Scalar-kernel handles continue to refer to the evaluator's current
hybridization cache; model/mesh lifetime and shared model metadata consolidation
remain required. No support regrouping, cutoff removal, or CUDA work occurred.

The additional [prepared evaluator test](test/c++/block_sparse/prepared_evaluator.cpp)
checks both special-line orientations on a density-interaction model, coefficient
slice values, endpoint statistics, matrix dimensions, and propagator ownership.
It binds an independently changed propagator and verifies scaled numerical views,
then overwrites walker metadata and verifies that a retained contribution keeps
its own copy. Existing analytic, partition, retarded-interaction, indexing,
correlator, and complex-basis tests validate the resulting arithmetic.

Validation: CPU configure/build, all 14 focused CTest entries, and all 70 CPU
CTest entries passed (full suite 56.62 s). Generated bindings are unchanged.
Logs: `/tmp/xca-backbone-operands-{configure,build,focused,test}.log`.
Next: finish shared model/kernel preparation and apply the planning/execution
boundary to correlator fragments and joins. Numerical/support policy and final
selection/adapter cleanup remain required; the persistent goal is active.


## Stage 5a — correlator fragments and joins (`eff9128`)

The correlator evaluator previously interleaved two block walks, their arithmetic,
and external-operator closure checks. It now has separate discovery, binding, and
execution functions. Read [correlator_plan.hpp](c++/triqs_xca/block_sparse/correlator_plan.hpp)
and its [implementation](c++/triqs_xca/block_sparse/correlator_plan.cpp), then
`eval_correlator_fragment`, `eval_correlator_join`, and their orchestrating
`eval_correlator` overload in
[diagram_evaluator.cpp](c++/triqs_xca/block_sparse/diagram_evaluator.cpp).

A fragment describes an inclusive propagator-edge interval, its start/end sectors,
and owned path/dimension metadata. A one-edge interval has no internal vertices
and remains a valid fragment. Binding supplies borrowed operator and propagator
views, using the same internal-vertex binder as self-energy. That binder moved
unchanged to [operator_binding.cpp](c++/triqs_xca/block_sparse/operator_binding.cpp).
The old block-index-consuming vertex/edge arithmetic adapters were removed.

The orchestrator executes fragments in their existing side/start-sector order
and retains the resulting matrices and end-sector maps for one assignment.
The join planner streams compatible right/left fragment pairs for each mu
operator and prescribes the closing kap operators and observable destinations.
Join arithmetic computes the shared left–mu–right product once, then its traces;
it makes no connectivity or closure decisions. Fragment results have their actual
matrix dimensions; unused sectors no longer preallocate padded result arrays.
No collection of all diagram assignments or all observable pairs is built.

Matrix/convolution order, the left-fragment reflection, prefactor placement,
external statistics, and mu/start/kap accumulation order are unchanged. Current
mask placement is deliberately retained: fragment walks inspect reached sectors,
and join discovery applies its old endpoint checks. There is no newly introduced
starting-edge check. Numerical/support policy is still a separate required stage.

Paths and join lists own their metadata; bound numerical views borrow the model,
current propagator, or external-operator inputs. They must be used with the matching
assignment/mesh before their owners change. Execution returns scratch views that
the orchestrator copies immediately into retained fragment results. No numerical
state is cached across propagator calls.

The new [correlator planning tests](test/c++/block_sparse/correlator_plan.cpp) use
an explicitly specified 1/2/1-dimensional sector ladder, independently of the
numerical evaluator. They check crossing paths, owned metadata after workspace
reuse, borrowed operands, reached-sector masks, and an empty fragment after the
other walk's final starting sector fails. First-order joins check rectangular
external operands, output indices, trace closure, absent fragments, and masks.
Existing shared correlator and analytic suites continue to validate values,
statistics, reflection, mixed interactions, and complex basis changes.

Validation: CPU configure/build, all 15 focused CTest entries, and all 71 CPU
CTest entries passed (full suite 56.68 s). Generated bindings are unchanged.
Logs: `/tmp/xca-backbone-correlator-{configure,build,focused,test}.log`.

Next: finish shared model/observable metadata and kernel preparation, then revise
numerical/support policy and consolidate selection interfaces. The block-specific
fragment/join extraction is complete, but the overall stage-3/5 common metadata
boundary, stage 6, final adapter removal, CPU timing checks, and completion audit
remain required. The persistent goal stays active.


## Shared scalar-kernel binding (`18d5cd5`)

Vertex/edge arithmetic still selected pole tables from `Hybridization`, and
block-sparse paths repeatedly computed the same scalar prefactor. The shared
[kernel_binding.hpp](c++/triqs_xca/kernel_binding.hpp) now describes these operands
explicitly. Read its [binder and numerical application](c++/triqs_xca/kernel_binding.cpp),
then the calls to `bind_kernels` at the prepared evaluator entry points.

`BoundKernels` borrows ordinary and reflected vertex-kernel views, holds ordered
lists of edge factors, and stores the scalar prefactor. Binding occurs once per
assignment. Block-sparse self-energy paths and correlator fragments reuse that
binding; dense ordinary/paired self-energy and correlators use the same interface.
Arithmetic applies supplied factors without reading pole indices or selecting
sign-dependent kernel tables. The old multiplication methods and diagram-assignment
dependency were removed from `Hybridization`, along with redundant evaluator
prefactor helpers.

Edge factors retain their pole order; they are not pre-multiplied into a new
numerical table. This preserves the existing operation order while allowing a
future executor to choose another materialization. Reflection is prepared for
vertex factors only, exactly as before; correlator edge-kernel and convolution
conventions remain unchanged. The scalar prefactor excludes the separate discrete
assignment, permutation, observable, and diagram-order signs.

Bound views require the original mesh/pole tables to remain alive and unchanged.
They are local to the prepared call, contain no propagator values, and must be
rebound after model/mesh replacement. A logical recipe can retain its pole indices
when pole magnitudes change without changing the sign-based placement rules, but
its numerical views cannot be reused for the replacement owner.

The additional [hybridization tests](test/c++/hyb.cpp) specify a negative-pole
vertex decomposition and an overlapping positive-pole edge decomposition directly
using scalar `cppdlr::k_it` values. They check reflected factors, absent-factor
identity behavior, ordered edge views, scalar prefactors, and independent bindings
to changed pole tables. Existing hand-derived preparation tests and analytic
whole-diagram suites remain intact.

Validation: CPU configure/build, all nine focused CTest entries, and all 71 CPU
CTest entries passed (full suite 56.91 s). Generated bindings are unchanged.
Logs: `/tmp/xca-backbone-kernels-{configure,build,focused,test}.log`.

Next: complete shared model-family and observable metadata, then numerical/support
policy and final selection-interface consolidation. The scalar-kernel boundary
now complements resolved matrix operands and block-specific path/join plans.
The legacy backbone adapters and cutoff/mask behavior still require their planned
cleanup; CPU timing checks and the full completion audit remain outstanding.

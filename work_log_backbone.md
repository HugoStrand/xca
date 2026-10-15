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
| Shared model/observable metadata | `5e7ab02` | Explicit families, flavour lists, and endpoint statistics; all 71 CTest entries passed |
| 6a | `0602db2` | Exact-zero masks and untruncated accumulation; all 72 CTest entries passed |
| 6b coefficient policy | `a7b1956` | Reject unrepresented cross-set coefficients; all 72 CTest entries passed |
| 6b operator support | `fa44a50` | Owned exact support and compatible retarded groups; all 72 CTest entries passed |
| 6b output closure | `3456087` | Validate indexed diagonal-output closure; all 72 CTest entries passed |
| 7a | `0a39662` | Production selection uses immutable topology; all 72 CTest entries passed |
| 7b | `71a7173` | Mutable backbone adapters removed; all 72 CTest entries passed |
| 7c | `dffa2b2` | Bounded selections and MPI integration; all 73 CTest entries and a three-rank run passed |
| Completion audit | None | Timing/memory checks and requirement audit pending |

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


## Shared model and observable metadata (`5e7ab02`)

The two evaluators previously reconstructed statistics from an implicit prefix
of fermionic flavours and selected single-particle external operators from the
same prefix. [diagram_model.hpp](c++/triqs_xca/diagram_model.hpp) now describes
interaction families, their statistics, and their logical flavour lists.
Its [implementation](c++/triqs_xca/diagram_model.cpp) validates that each flavour
belongs to exactly one family and preserves explicit external flavour order.
A flavour continues to identify its operator and scalar-coefficient axes; the
metadata owns no matrix, pole, mesh, or block-symmetry data.

Read those files, then the `interactions_` members and single-particle operator
selection in the dense and block-sparse evaluators. Both consume the model's
statistics for sign preparation and its hybridization flavour list for external
single-particle operators. The current atom-diag builder still stores hybridization
operators before retarded ones, but this layout is translated by
`hybridization_retarded_model` once per evaluator. The legacy statistics-vector
helper delegates to that adapter. Metadata itself permits noncontiguous flavour
lists; this change does not introduce a new public model-construction API.

`ObservableBoundary` names the statistics at zero and tau. Prepared correlator
entry points now take that boundary instead of a boolean, and both evaluators
apply the prepared endpoint sign using it. Existing backbone/indexed entry points
translate their uniform-statistics boolean through `uniform_observable`.
The public many-body-operator API retains its validation of homogeneous operator
statistics; expanding that API is not part of this structural change. Prepared
C++ tests were updated explicitly, while old direct-backbone statistics tests
continue to exercise the compatibility route.

The new model tests in
[interaction_contractions.cpp](test/c++/interaction_contractions.cpp) cover owned
metadata, interleaved bosonic/fermionic flavours, external order `{3,1}`, missing,
duplicate, and out-of-range membership, and the atom-diag adapter. A hand-specified
crossing diagram checks fermionic, bosonic, and mixed endpoint boundaries against
known permutation signs. Existing single-particle tests with retarded operators
check that the physical output excludes retarded-interaction flavours.

Validation: CPU configure/build, all nine focused CTest entries, and all 71 CPU
CTest entries passed (full suite 56.61 s). Generated bindings are unchanged.
Logs: `/tmp/xca-backbone-model-{configure,build,focused,test}.log`.

This supplies the remaining common model/observable descriptions for the stage-3
and stage-5 extractions. Shared preparation, block-specific path/join planning,
and numerical consumers are connected; final consolidation still requires removal
of the legacy backbone and selection scaffolding.

Next is stage 6, in two review units: numerical zero policy, then support/grouping
semantics. The initial audit found magnitude masks in `BlockDiagOpFun` import and
mask refresh, the self-energy contribution cutoff, and a similar mask in the
atomic-propagator test helper. Independent scaled/cancellation tests must avoid
inheriting those approximations. Cross-set coefficient tolerance remains a
separate support-policy issue. Stage 7, CPU timing checks, and the requirement-by-
requirement completion audit remain outstanding; the persistent goal is active.

## Stage 6a — exact numerical zeros (`0602db2`)

The old `1e-16` policy affected three different boundaries: importing a propagator,
refreshing an accumulated block's mask, and admitting a self-energy contribution.
A small input could disappear before traversal, and a cancellation residual could
be overwritten by the next addition because the accumulator was marked zero.
This commit deliberately changes that policy after the structural extractions.

Read [numerical_zero.cpp](test/c++/block_sparse/numerical_zero.cpp) first, then
[block_op.cpp](c++/triqs_xca/block_sparse/block_op.cpp) and
`DiagramEvaluator::eval_self_energy_paths`. Imported and refreshed masks now mark
a block zero only if every complex entry equals zero exactly. Direct equality
also preserves subnormal entries without computing their magnitudes. Every valid
self-energy path is accumulated, without a magnitude cutoff. Explicit caller
masks still assert exact structural zeros, including absent storage; a block
marked active may conservatively contain zeros. Ordinary floating-point rounding
and DLR approximation remain unchanged.

The new tests use raw tensors and closed-form first-order one-fermion results,
not masked full-Hilbert-space converters. They check dense and block-sparse
self-energies and single-particle functions for normal inputs, propagator scale
`1e-18`, coefficient scale `1e-18`, combined scaling, and exact-zero controls.
The nonzero analytic comparisons use relative error `2e-11` at DLR tolerance
`1e-12`, with no absolute floor that could hide lost contributions. Separate
tests cover subnormal mask entries and exactly representable binary cancellation,
followed by both `add_block` and `operator+=` and eventual exact cancellation.
All three new tests failed against the prior implementation; normal-scale
analytic controls passed.

The reference-helper audit found the same cutoff in `get_hamiltonian_blocks`;
it now detects exact zeros too. Manual block-sparse routines use the revised
container masks, while the dense conversion helper respects caller masks.
The full suite exposed one old `block_diag_op_fun` assertion treating `1e-17`
as zero. Its expectation now explicitly requires preservation of that value.
No comparison tolerances were relaxed. Generated bindings are unchanged.

Validation: CPU configure/build, six focused CTest entries, and all 72 CPU CTest
entries passed (full suite 56.62 s). The first full run passed 71 entries
and failed only the legacy mask-policy assertion described above.
Logs: `/tmp/xca-backbone-zero-{configure,build,focused,test-final}.log`;
pre-change failures: `/tmp/xca-backbone-zero-reproduce.log`.

Next is stage 6b: distinguish individual operator support from group candidate
paths, characterize/reject or retain cross-set couplings explicitly, and validate
physical partition assumptions. The existing cross-set tolerance and equal-set
path restriction are still present and are not justified by this numerical-zero
change. Stage 7 adapter/selection cleanup, CPU timing checks, and the completion
audit remain outstanding. The persistent goal is active.

## Stage 6b — cross-set coefficient policy (`a7b1956`)

This is the first support-policy review unit. Before changing grouping, it resolves
what the previous `1e-12 * max|coefficients|` exception actually discarded.
Read the closed-form two-fermion test in
[sym_set_mask.cpp](test/c++/block_sparse/sym_set_mask.cpp), then
`check_sym_set_block_diagonal` in
[block_op.cpp](c++/triqs_xca/block_sparse/block_op.cpp).

In the explicit Fock basis `|00>, |10>, |01>, |11>`, a zero-frequency pole gives
`Delta=-C/2`. With a cross-flavour coefficient `delta`, the first-order
self-energy has `Sigma(1,2)=-delta/2*(G00-G33)`. Constant diagonal propagators
`G00=s`, `G33=3s` therefore give `Sigma(1,2)=delta*s`. The test uses
`delta=1e-14`, below the old acceptance threshold, and scales `s=1` and `1e8`.
Dense and coarse-block evaluators both reproduce the nonzero `1e-14` and `1e-6`
entries within relative error `2e-11`, without an absolute tolerance floor.
A representation with four separate one-state sectors cannot store that entry.
Thus a small coefficient is neither an exact structural zero nor a general
bound on diagram error. This example passed before the guard change too.

The current grouped bars contract only within one group. The guard now rejects
all nonzero cross-group coefficients, including imaginary subnormal values,
with a message identifying the coefficient and groups and suggesting compatible
grouping or a coarser physical partition. Representable within-group terms have
no new magnitude restriction. Both quartet construction and the evaluator's
quartet-only constructor apply the guard. The obsolete public tolerance constant
is removed. This explicitly rejects unsupported input instead of returning a
silently truncated model; it does not claim that every rejected grouping is a
fundamental physical restriction.

The same audit found that the old `std::max` reduction did not detect NaNs as
intended. Finiteness is now checked on each real and imaginary component, within
and between groups. The tests exercise those cases, zero cross-set controls,
coefficient scales `1` and `1e6`, and unchanged sensitivity to tiny allowed
within-set terms. Prior tests requiring a tiny cross-set perturbation to leave
Sigma identically unchanged have been replaced by explicit rejection checks.
The new rejection/NaN checks failed before the implementation change.

Validation: full CPU build, all six focused CTest entries, and all 72 CPU CTest
entries passed (full suite 56.94 s). Generated bindings are unchanged.
Logs: `/tmp/xca-backbone-support-{build,focused,test}.log`;
pre-change evidence: `/tmp/xca-backbone-support-reproduce.log`.

The rest of stage 6b remains required: individual exact operator support must
be distinct from candidate group connectivity, compatible density operators with
different support must be groupable without epsilon terms, and physical partition
closure must be validated. General operators can need multiple block components;
the single-target numerical storage must not silently redefine their support.
Stage 7 selection/adapter removal, CPU timing checks, and the final audit are
also pending. The persistent goal remains active.

## Stage 6b — operator support and compatible groups (`fa44a50`)

Previously a group's connection row also stood in for each member's support.
This forced density operators with different zero sectors into separate groups,
so their cross-coefficients could not be represented even though all their
nonzero blocks are diagonal. The new boundary separates a candidate group
schedule from the actual support of each materialized operator.

Read [operator_support.hpp](c++/triqs_xca/block_sparse/operator_support.hpp) and
its implementation, then the grouping loop in
[dynint.cpp](c++/triqs_xca/block_sparse/dynint.cpp), followed by the two path
planners and `bind_self_energy_contribution`. `OperatorSupport` owns source/target
connectivity in a compact offset/target representation. It can describe multiple
targets per source; its `single_target` adapter explicitly rejects that case for
the current CPU executor. This does not add multi-target numerical storage or
remove the atom-diag builder's existing rejection of operators that cannot be
represented by its one-block-per-column storage.

`OperatorSupports` describes plain, adjoint, barred-adjoint, and reflected-barred
operators independently, including each pole. It detects exact zero matrices
once after materialization. A bar may have support where its retained plain
operator vanishes, so its support is never inferred from that operator. Metadata
is owned and contains no propagator values. The evaluator constructs it once
from its private quartet; an external planner caller must rebuild it after
replacing or mutating that quartet. There is no matrix scan per indexed call.
Storage scales with model sectors, flavours, and poles, not the number of diagram
assignments. Timing/memory checks remain part of final consolidation.

Internal self-energy and correlator traversal now uses the selected member's
support. Candidate group maps still provide dimensions and special-endpoint path
candidates; endpoint binding selects the individually present operators before
numerical execution. Remaining zero coefficient combinations can still produce
zero contributions, but no small nonzero block is classified as absent.

Retarded operators are grouped in first-appearance order when the union of their
forward and adjoint maps remains single-target. Both orientations are checked,
so the union is injective. Storage covers that candidate union and zero-fills
missing member blocks; no physical operator is changed and no epsilon is added.
Fermionic grouping remains unchanged. This is a greedy compatible grouping,
not a search for an optimal partition. Nonzero couplings between resulting
incompatible groups are still rejected explicitly by the preceding commit.

Tests specify density occupation values directly in four one-state sectors and
check their zeros, shared group, and the changed support of contracted bars.
Additional checks cover owned/multi-target metadata, tiny operator values,
coefficient replacement, and an internal path whose plain and barred supports
are disjoint. The synthetic planner fixtures now contain nonzero operator blocks
on the intended paths; their existing connectivity expectations remain intact.

A whole-evaluator test adds asymmetric complex couplings between two density
operators on a diagonal two-fermion Hamiltonian. Self-energy, single-particle
functions, and density correlators agree with the dense one-block model for
orders one through three within relative `2e-10`. Existing analytical suites
remain unchanged. The only full-suite failure during migration was an assertion
that single-orbital `S+` and `S-` must occupy different groups. Their supports are
disjoint and their union is a valid swap; the revised test checks their individual
support against atom_diag and still requires separate groups for the conflicting
two-orbital maps. No numerical tolerance was relaxed.

Validation: CPU configure/build, five focused CTest entries, and all 72 CPU CTest
entries passed (full suite 56.19 s). Generated bindings are unchanged.
Final logs: `/tmp/xca-backbone-group-build-verified.log` and
`/tmp/xca-backbone-group-test-final.log`; focused checks:
`/tmp/xca-backbone-group-focused-final.log`.

Next: complete the physical-partition closure check at the orchestration boundary,
then stage 7's selection interfaces and removal of legacy backbone adapters.
CPU timing/memory checks and the requirement-by-requirement completion audit
remain outstanding. The persistent goal remains active.

## Stage 6b — self-energy output closure (`3456087`)

Path discovery records both the starting/output sector and terminal sector, but
previously the evaluator accumulated every result into the starting diagonal
block. Shape checking caught unequal dimensions; equal-sized sectors silently
accepted a matrix belonging to an off-diagonal block.

Read the new `nonclosing_paths_require_a_coarser_output_partition` test in
[prepared_evaluator.cpp](test/c++/block_sparse/prepared_evaluator.cpp), then
`select_diagonal_self_energy` in
[self_energy_contribution.cpp](c++/triqs_xca/block_sparse/self_energy_contribution.cpp)
and its call in `eval_self_energy_paths`. The selector belongs to orchestration,
after binding and before numerical execution. Discovery can still describe
rectangular or nonclosing paths for another consumer; the arithmetic executor
has no output-partition decisions.

The regression uses self-adjoint permutations `A=(01)` and `B=(12)` on three
one-state sectors. Each interaction separately closes at first order, but the
crossed word `BABA` cycles between sectors. For constant identity propagators,
zero-frequency scalar kernels, and a fixed second-order bosonic assignment, the
coarse-block answer is `-tau^2/8 * (BABA + I)`. The test checks every entry against
that formula, including off-diagonal values larger than `0.1`. Before this change,
that analytic check passed while the fine-partition rejection check failed:
the fine evaluator silently accumulated into the wrong diagonal blocks.

The diagonal-output consumer now requires each active indexed contribution to
close. On a nonclosing candidate, it inspects only the already-bound input
coefficient views: if all endpoint couplings are exactly zero, the candidate is
skipped; otherwise it reports the source/terminal sectors and requests a coarser
partition. No magnitude threshold or computed-result decision is introduced.
A second control removes the A coupling and checks the remaining B-only result,
so unused candidate connectivity cannot cause a false rejection. The test also
checks a valid first-order call after rejection to detect stale accumulation.

This is a structural requirement on the current indexed, block-diagonal output
API. It does not attempt to establish diagonal output through cancellations
between distinct contributions, nor prove that every structurally possible
matrix product is numerically nonzero. Supporting such cases would require a
consumer that can represent the intervening off-diagonal work. General-correlator
joins already prescribe trace closure and are unchanged.

Validation: full CPU build and all 72 CPU CTest entries passed (56.16 s).
Generated bindings are unchanged. Logs:
`/tmp/xca-backbone-closure-{build,test}.log`; pre-change regression:
`/tmp/xca-backbone-closure-reproduce.log`.

This completes stage 6's numerical/support-policy units: exact numerical zeros,
explicit handling of unrepresented coefficient couplings, distinct member/group
support, compatible groups, and output-partition validation. Stage 7 remains:
consolidate selection interfaces, preserve MPI coverage and paired-direction
semantics, remove the legacy mutable backbone adapters, and finish CPU timing/
memory checks and the final requirements audit. The persistent goal is active.

## Stage 7a — production selection without mutable backbones (`0a39662`)

The numerical routines already consumed prepared assignments, but public
self-energy/correlator calls still constructed mutable `Backbone` objects to
carry a topology and obtain counts. Production selection now takes
`DiagramTopology const &` and calls the shared count/assignment preparation
functions directly. Read the selection overloads in
[dense/diagram_evaluator.cpp](c++/triqs_xca/dense/diagram_evaluator.cpp), then the
corresponding block-sparse overloads. Neither evaluator includes `backbone.hpp`
or constructs a `Backbone` or `CorrelatorBackbone` anymore.

This is a data-flow refactoring. Public Python scalar/vector/all-assignment calls,
index order, exceptional-input checks, and accumulation order remain unchanged.
Dense paired evaluation still visits only the seeds whose first direction bit
is zero in its all-assignment overload; explicit scalar/vector selections still
evaluate both partners of each supplied index. Its direction stride is derived
from the shared counts rather than a mutable backbone field. Interaction and
observable statistics continue to come from the already-separated model/boundary
metadata.

The C++ topology selection overloads now require an explicitly constructed
`DiagramTopology`. The constructor is marked `explicit` to prevent ambiguity
between NDA array-to-view conversion and conversion to topology. Existing array
APIs continue to accept NDA arrays normally. Prepared numerical overloads are
unchanged. These topology overloads are still excluded from Python wrapping.

The two public debug-print methods keep their names and index validation, but
print the immutable assignment directly. Its diagnostic lists topology,
directions, poles, vertex recipes, edge factors, and scalar-prefactor recipes.
This removes the last production need for the legacy mutable printer; the text
layout is deliberately changed rather than retaining an adapter solely for it.
The generated Python wrapper changed only in the two evaluator class docstrings,
which now describe immutable assignments instead of a Backbone object.

Direct evaluator tests for sparsity invariance, the empty correlator fragment,
and forced observable statistics now pass an immutable topology. Their numerical
comparisons and negative controls are unchanged. The prepared block-sparse test
compares its logical entry point with the topology/index entry point. The direct
legacy indexing/parity tests, one sign-sensitivity helper, and the source-backbone
mutation test still use the old classes and are the next migration unit; they
must be rewritten around their semantic assertions before the classes are deleted.

Validation: full CPU build and all 72 CPU CTest entries passed (56.92 s).
Logs: `/tmp/xca-backbone-topology-{build,test}.log`. The build caught and resolved
the NDA overload ambiguity described above; the last source edits after compilation
only removed duplicated words in comments. No numerical or test tolerance changes.

Next: remove the remaining direct-test dependencies and legacy implementation,
then consolidate bounded range selections and MPI/paired coverage. Targeted CPU
timing/memory checks and the final completion audit remain required. The goal
remains active.


## Stage 7b — remove mutable preparation adapters

Code: `71a7173` (`[backbone] remove mutable preparation adapters`).

The production evaluators already consumed owned assignments. This commit moves
its remaining test clients and deletes `backbone.hpp/.cpp`, including mutable
vertex setters, reset sequencing, external-statistics sentinels, and duplicate
counting/printing interfaces. The redundant `fermion_boson_statistics` adapter
is also gone; tests use the explicit interaction model catalogue.

Read `test/c++/backbone_indexing.cpp` first: both logical and flat selections now
run against the existing hand-derived third-order expectations for every line
direction. Large counts, mixed digits beyond `INT_MAX`, overflow, invalid indices,
zero poles, and owned-copy checks remain. The old reset/reuse test becomes an
independent-preparation test: vertex-kernel and edge-kernel assignments retain
their different recipes after later preparation, including a rejected request.

Then read `test/c++/backbone_parity.cpp`. The complete hand-derived parity table
and the deliberately incorrect all-fermionic catalogue remain. Explicit
observable statistics replace the old -2/-3 sentinels. The old rejection of an
unresolved *external* orbital is intentionally removed: external endpoints are
now resolved separately. The existing explicit-statistics test still rejects an
unset *internal* orbital and verifies that unresolved external endpoints work.
The correlator integration test retains its independently computed filtered
topology parity, rather than borrowing the production endpoint-sign result.
Finally, the path-discovery ownership test now mutates a separate source
assignment while visiting a retained copy.

Test filenames remain unchanged so their history and CTest labels are easy to
follow. Comments referring to the deleted API were updated. Numerical evaluator
algorithms and tolerances did not change; generated bindings did not change.

Validation: CPU reconfiguration, full rebuild, and all 72 CTest entries passed
(56.40 s). Logs: `/tmp/xca-backbone-adapter-{configure,build,test}.log`.
An include/type search finds no remaining uses of the removed classes in source,
Python, or tests. Next are bounded ranges and MPI/paired coverage, followed by
CPU timing/memory measurements and the completion audit. The goal stays active.


## Stage 7c — bounded assignment selections and MPI coverage

Code: `dffa2b2` (`[backbone] stream bounded assignment ranges`).

Previously each Python rank allocated its whole local flat-index vector, and
filtered that vector again for dense paired evaluation. Both CPU evaluators now
accept half-open bounds `(first, last)` for self-energy, single-particle Green's
functions, and general one-time correlators. The solver passes two integers,
using the same contiguous `np.array_split` partition. `AssignmentSelection`
streams indices into the existing logical preparation boundary, with constant
selection storage. Existing explicit vector calls borrow their input for the
call; their order and multiplicity remain meaningful.

Read [diagram_selection.hpp](c++/triqs_xca/diagram_selection.hpp), then the
`compute_*_selected` methods and their range/vector wrappers in each evaluator.
These wrappers share the existing numerical path; they do not introduce a second
evaluator. Full public self-energy/single-particle calls use the same selection
loop. Prepared C++ assignment entry points remain independent of linear ranges.
Then read `scatter_range_over_ranks` and the four solver call sites. The helper
now returns bounds rather than an index array: this is an intentional Python
helper API change. Evaluator scalar/vector APIs remain available.

Dense paired **ranges** own only canonical seeds whose first direction bit is
zero. Their partner can lie outside the supplied bounds. Consequently disjoint
ranges covering the full self-energy index space count every pair once. Existing
scalar/vector paired APIs still evaluate both partners of every supplied seed,
including duplicates. This distinction is stated on the range declaration and
in the solver. Empty ranges return correctly shaped zeros; invalid bounds are
rejected before evaluation. Long bounds support selections beyond `INT_MAX`.

The one-time-correlator template range overloads are instantiated for both real
and complex `atom_diag`, including the explicit declarations used by the Python
wrapper generator. Generated bindings are committed alongside the API change.
The first focused Python run caught missing template-wrapper declarations;
these were supplied before the final validation.

`flat_index_range.cpp` checks partition sums against the existing vector path
on complex, mixed fermionic/retarded fixtures at orders two and three, for both
CPUs and all observable types. It covers empty partitions, invalid/reversed
bounds, recovery, large actual evaluation indices, and traversal next to
`LONG_MAX`. Dense pair sums are compared with unpaired self-energy. The Python
`range_selection.py` checks integer partitions, including zero/huge counts, and
binding/reduction equivalence. A three-rank run exercises real MPI, including
more ranks than first-order assignments. No numerical tolerances were relaxed.

Validation: CPU configure/build and all 73 CTest entries passed (59.36 s).
Logs: `/tmp/xca-backbone-ranges-{configure,build,test}.log`.
The standalone MPI check passed with:

```bash
source ~/env/xca-cpu.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/cbuild/python:$PWD/cbuild/deps/adapol_src/src:$PWD/cbuild/deps/pyed_src:$PYTHONPATH"
mpiexec -n 3 python3 test/python/range_selection.py
```

Its log is `/tmp/xca-backbone-ranges-mpi.log`. This run is additional local
validation; CTest registers the same script in serial. Remaining work is the
planned CPU preparation/memory measurements and final requirement audit.

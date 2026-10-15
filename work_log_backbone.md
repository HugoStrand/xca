# Backbone refactoring work log

## Review index

| Stage | Code commit | Status |
| --- | --- | --- |
| Planning | None | Review process and staged draft recorded |
| Test dependency review | None | Backbone coupling inventoried in the plan |
| Plan review | None | Revised implementation plan ready for execution |
| 0 | None | CPU configure/build and all 66 CTest entries passed |
| 1–7 | None | Not started |

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

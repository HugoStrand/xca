# Backbone refactoring work log

## Review index

| Stage | Code commit | Status |
| --- | --- | --- |
| Planning | None | Review process and staged draft recorded |
| Test dependency review | None | Backbone coupling inventoried in the plan |
| Plan review | None | Revised implementation plan ready for execution |
| 0–7 | None | Not started |

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

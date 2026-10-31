# Backbone reversal work log

2026-10-30. Branch dev_backbone; source baseline 522ad52.
Plan: plan_reverse_from_diagram_to_backbone.md.

Current status (2026-10-31): implementation, adversarial review and validation
complete. Source checkpoints are listed at the end; documentation commits are
separate. No push or benchmark measurements.

The requested design makes enriched Arcs independent preparation inputs,
Backbone the complete dense instruction description, and one owning BackbonePath
the complete sparse description for self-energy and correlator work. Numerical
Action ownership and the functioning Python APIs remain unchanged.

Initial unrelated state: modified .gitignore; untracked AGENTS.md,
AGENTS.md.bak-20261009, test/c++/test_nda_cuda_smoke.cpp,
test/python/__pycache__/, test/python/exp_order_conv/ and tools/. Preserve all.

Plan reviewed independently by three agents (shared diagram preparation, sparse
execution, Action/F-bar boundary) and revised. The main clarifications concern
metadata validity, numerical-owner compatibility, propagator-mask rediscovery,
and exactly-once correlator normalization with cached segments. Interface-changing
checkpoints must also keep the benchmark consumer compiling.

Initial status: reviewed plan ready. Source implementation starts with numerical
operator conventions and enriched arc metadata.

## Checkpoint 1 — operator conventions and enriched arcs

2026-10-31. Source commit: [action] separate operator conventions.

The F-bar partial-contraction and operator-pair orientation descriptions now live
in action.hpp, beside the numerical data they describe. Diagram vertex preparation
uses direction flags directly. The old endpoint-sum descriptor is a temporary
bridge and will disappear when endpoints move into Backbone.

Arc preparation now resolves statistics and three-way pole signs from explicit
metadata or Action, validates selected indices, and replaces stale derived fields.
The existing flat digit order and numerical contraction formulas are unchanged.

Validation: CPU reconfigure and focused build passed; diagram/arcs (six cases),
interaction_contractions, backbone_indexing and backbone_parity all passed.
No Python binding changes; source diff --check clean. Logs:
/tmp/xca-backbone-reverse-1-{configure,build}.log.

Final plan-review clarification: full-path execution must reject selected ranges;
empty endpoint lists never imply identity and observable cardinality is checked.

Next: resolve Backbone instructions and migrate dense/sparse consumers, preserving
the specialized sparse completion wrappers only through the transition.

## Checkpoint 2 — self-contained Backbone

2026-10-31. Source commit: [diag] evaluate prepared backbones.

The runtime Diagram wrapper and diagram/arc.hpp are gone. Preparation now follows
Action metadata -> enriched Arcs, then Topology + Arcs -> Backbone. Vertices and
kernel factors carry resolved flavour/pole indices. Backbone owns endpoint
geometry, both permutation-sign cases, internal direction sign and ordered scalar
normalization terms. No evaluation lookup needs Topology or Arcs.

Dense numerical entry points consume Backbone; ordinary/barred FSet access is
numerical and no longer depends on diagram headers. Native dense correlators
borrow observable arrays. Sparse discovery and its temporary binding layer also
consume Backbone, preparing the next path-ownership change.

This checkpoint changes both evaluators together because removing Diagram and
arc-relative kernel instructions changes their shared interfaces. The sparse
completion records and numerical-view binding remain deliberately unchanged in
structure; their removal is a separate checkpoint.

Tests retain literal indexing/parity fixtures and now prove prepared dense
self-energy/pair/correlator evaluation after construction inputs are destroyed.
Repeated-pole normalization is checked against a finite arithmetic example that
would overflow if factors were regrouped. Independent cross-reviews of dense and
sparse numerical migration found no actionable regression.

Validation: full CPU build and backbone_cpu compilation passed; all 77 CPU/Python
CTest entries passed in 57.51 s. Generated Python binding changes are docstrings
only. No benchmark measurements were run. Source diff --check clean. Logs:
/tmp/xca-backbone-reverse-2-{configure,build,test,benchmark-build}.log.

The next checkpoint uses one generic owning BackbonePath for segments and complete
contributions. Discovery will own one streamed workspace per Backbone call rather
than expose a caller workspace that initially has no recipe. This simplifies the
API, at the cost of no longer amortizing its small array allocations across
different assignments; no performance claim is made for that tradeoff.

## Checkpoint 3 — one owning sparse path

2026-10-31. Source commit: [bs] prepare self-contained backbone paths.

BackbonePath now owns a Backbone, selected range, sector/dimension arrays and
ordered endpoint selections. SelfEnergyDiagram and CorrelatorDiagram are removed.
Self-energy discovery resolves exact supported endpoint flavour IDs in the same
group/member order as before and owns its reusable callback workspace.

Correlator completion creates a full path containing both segment walks and the
closing sector. Cache indices accompany the callback only; they are absent from
the path and its temporary numerical-view operands. The CPU still evaluates each
segment once. Existing nonclosing self-energy checks and numerical accumulation
order remain unchanged. Binding now derives all logical information from Path;
its removal is the next independently reviewable change.

Geometry tests cover full slot/dimension composition, source/table destruction,
copy independence, empty segment tables, exact endpoint support and interleaved
global flavour labels. The obsolete wrong-workspace-order case disappears with
the caller-supplied workspace API.

Validation: full CPU build and backbone_cpu compilation passed; all 77 CPU/Python
CTest entries passed in 57.02 s. Generated binding delta is an updated class
docstring only; source diff --check clean. No benchmark measurements. Logs:
/tmp/xca-backbone-reverse-3-{configure,build,test,benchmark-build}.log.

Next: direct Path execution, removal of binding/operand containers, and standalone
numerical comparisons after all planning/cache owners have been destroyed.

## Checkpoint 4 — direct sparse path execution

2026-10-31. Source commit: [bs] evaluate backbone paths directly.

Sparse numerical routines now read operator matrices, propagator blocks and
endpoint coefficients directly from Action/G using BackbonePath identifiers.
The bind directory and BackboneOperands, SelfEnergyOperands and CorrelatorOperands
containers are removed. Numerical owners are borrowed only during execution;
observable vectors are passed by const reference throughout the native chain.

The new native full-path entry points execute either endpoint operation without
rediscovery. Self-energy preserves the exact nonclosing-path coefficient check,
endpoint accumulation order and block-diagonal result policy. Correlator execution
can slice a retained full path into its two local traversals. It uses separate T/U
scratch and the same completion routine as cached evaluation. Normalization and
internal direction factors are applied once to the before-special result;
external permutation parity is applied after accumulation.

The Backbone-level correlator API still discovers and evaluates each segment once
and reuses its result across observable joins. Cache indices remain local to that
execution strategy. Full paths do not store or require them.

Four new numerical tests exercise retained paths after topology/arcs/backbone,
discovery workspaces and segment tables are destroyed. They cover first and
crossed second order, both self-energy directions, multiple joins sharing cached
segments, non-unit normalization, physical fermionic observables with odd crossing
parity, and changed-propagator scaling. Wrong-role/incomplete paths and multiple
tau selections are rejected before scratch mutation; empty selections give zero.
The removed binding-specific pointer tests are replaced by these execution tests.

The benchmark source is adapted for compilation only. Its obsolete binding phase
is removed and the README distinguishes stored results for that former stage.
No timing measurements were made.

Validation: full CPU reconfigure/build and backbone_cpu compilation passed. All
78 CPU/Python CTest entries passed in 57.11 s, including all four cases in
block_sparse/path_evaluator. Binding regeneration produced no additional changes
at this checkpoint; Python-facing signatures remain unchanged. Diff --check
passed. Logs: /tmp/xca-backbone-reverse-4-{configure,build,benchmark-build,test}.log.

## Final adversarial review

Independent read-only reviews covered direct sparse execution and the shared
preparation/Action boundary. No blocking correctness findings remained. Checks
included exact coefficient axes/reflection signs, rectangular path geometry,
normalization placement, separate traversal scratch, cache independence, selected
range rejection, zero-pole placement and repeated kernel-factor ordering. Literal
flat-index/parity fixtures and independent manual numerical oracles remain intact.

Deliberate limits and tradeoffs:

- A retained path owns a full Backbone copy and sector metadata. No numerical
  matrices are copied into it. Discovery can stream one workspace; constructing
  all paths in advance is not required.
- The current CPU self-energy routine accepts the endpoint list from a discovered
  symmetry group or a subset; its scratch-capacity bound is checked. Correlator
  completion accepts one tau observable and an ordered closing subset. Neither
  restriction requires a separate path class or constrains a future GPU scheduler.
- Prepared objects assume compatible Action indices, statistics, pole signs and
  sector/basis layout. Individual paths can execute with changed G values; a
  saved path set needs rediscovery if G's activity mask changes. Forged public
  aggregates and cross-model compatibility are outside this refactoring's checks.
- Numerical work and public result/accumulation behavior are preserved. The
  benefit established here is the simpler preparation/execution boundary; no
  performance improvement or GPU execution claim is made.

## Commit reading guide

Read each source commit with the immediately following documentation commit.

| Source commit | Main change | Work-log section |
| --- | --- | --- |
| `0ed77bd` `[action] separate operator conventions` | Separate numerical F-bar conventions; enrich arc inputs | Checkpoint 1 |
| `912cdb4` `[diag] evaluate prepared backbones` | Resolve instructions in Backbone; remove Diagram | Checkpoint 2 |
| `e8af49e` `[bs] prepare self-contained backbone paths` | Own sparse geometry and endpoint selections in one path type | Checkpoint 3 |
| `1e565e0` `[bs] evaluate backbone paths directly` | Remove bind/operand layers; execute retained paths directly | Checkpoint 4 |

The initial plan is `8b188e8` `[docs] plan self-contained backbones`. All work is on
dev_backbone above the user's `522ad52` header consolidation. Unrelated worktree
changes were preserved.

## Follow-up — one preparation call in evaluators

2026-10-31. Source: `e0852f7` `[diag] simplify backbone preparation`.

Dense and sparse evaluators now call
`prepare_backbone(topology, action, flat_index, kind)`. The overload composes the
existing arc decoding and backbone preparation; all nine evaluator call sites
use it. As requested, this is an evaluator API simplification: the lower-level
`prepare_backbone(topology, arcs)` and arc APIs remain available, and tests and
benchmarks are unchanged.

Independent review found no issues with overload resolution, ownership or argument
preservation. The library and both focused test targets built successfully;
dense/prepared_evaluator and block_sparse/path_evaluator passed (2/2 CTest entries).
Diff --check passed. Logs: /tmp/xca-backbone-prepare-api-{build,test}.log.

## Follow-up — consolidate correlator helpers

2026-10-31. Source: `d604d5c` `[diag] consolidate correlator helpers`.

CorrelatorStatistics and uniform_correlator_statistics now live in backbone.hpp
and backbone.cpp with their existing names, defaults and behavior. Statistics
remain a separate per-call observable argument; Backbone gains no new state.
Removed diagram/correlator.hpp and diagram/correlator.cpp, including the redundant
topology-only correlator_backbone_ranges helper.

Its former test callers now prepare a Backbone and use endpoint_ranges(), keeping
the literal range expectations. The legacy-to-current comparison is replaced by
an explicit before-special range check. Includes are updated throughout the
evaluators and tests. Independent review confirmed that evaluator signatures,
statistics behavior and independent geometry expectations are preserved.

Validation: CPU reconfigure and full build passed; all 78 CPU/Python CTest entries
passed in 57.02 s. Binding regeneration produced no changes, no references to the
removed files/helper remain in source consumers, and diff --check passed. Logs:
/tmp/xca-correlator-cleanup-{configure,build,test}.log.

## Follow-up — vertex-based dense operator lookup

2026-10-31. Source: `7c5ac64` `[dense] fetch operators by vertex`.

FSet now provides get_operator(diagram::Vertex const&), selecting the ordinary or
barred matrix from the resolved vertex and returning a const view into its own
storage. Both dense traversal directions use the getter and pass the same vertex
to apply_kernel. Scalar accessors remain for F-bar construction and endpoint sums.
This deliberately adds a small lookup dependency on Vertex: fset.hpp only forward
declares it, and fset.cpp includes backbone.hpp. Construction arithmetic is unchanged.

Independent review found no view-lifetime, overload or traversal issues. The
library and four selected test targets built successfully; dense/prepared_evaluator,
interaction_contractions, one_fermion_se_spgf_all_evals and
one_time_correlator_all_evals passed (4/4 CTest entries, 3.48 s). These include
independent closed-form one-fermion checks through third order, complex F-bar
contraction checks and dense/sparse correlator comparisons. No new tests were
needed for the delegated lookup. Diff --check passed. Logs:
/tmp/xca-fset-vertex-{build,test}.log.

## Follow-up — derived endpoint orientation

2026-10-31. Source: `a71ac04` `[diag] expose endpoint orientation`.

Backbone::endpoint_orientation() derives OperatorPairOrientation from the existing
endpoint direction and rejects correlator endpoints through require_self_energy().
OperatorPairOrientation stays in action.hpp for independent F-bar construction
and gains opposite(), which returns the reversed value without changing its source.

Dense self-energy and paired evaluation now obtain the orientation from Backbone;
the pair uses orientation.opposite() for its second endpoint contraction. Sparse
endpoint multiplication and exact nonclosing coefficient checks use the same
accessor. No orientation state is duplicated. The pair retains T - U and applies
the original Backbone's kernel prefactor and global sign exactly once.

Independent review found no sign, wrong-role rejection or scope issues. Existing
wrong-role tests check rejection before scratch changes; numerical pair and
sparse nonclosing-path tests cover the affected behavior.

Validation: the library and all seven selected test targets built successfully.
diagram/backbone, interaction_contractions, dense/prepared_evaluator,
block_sparse/prepared_evaluator, block_sparse/path_evaluator,
one_fermion_se_spgf_all_evals and flat_index_all_evals passed (7/7 CTest entries,
5.27 s). This includes independent one-fermion reference checks through third
order. Diff --check passed. Logs: /tmp/xca-endpoint-orientation-{build,test}.log.

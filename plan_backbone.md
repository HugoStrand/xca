# Backbone and CPU evaluator refactoring

Status: implemented and validated, 2026-10-15.
The completion audit and remaining limitations are recorded at the end of
`work_log_backbone.md`.
Base: `dev_backbone`, branched from `origin/dev_hs` commit `2b0fc5c`.
Progress, commit identities, and validation evidence live in `work_log_backbone.md`.

## Objective and boundaries

Separate diagram preparation, representation-specific planning, and numerical
execution by refactoring the existing working CPU evaluators incrementally.
The shared diagram layer must serve dense and block-sparse evaluation without
requiring dense evaluation to discover sector paths. Block-sparse planning adds
sector connectivity, dimensions, observable closure, and output destinations.
CPU and future GPU execution choose their own storage, batching, and arithmetic.

The numerical objective is the accurate intended diagram sum, not bitwise
agreement with the old CPU evaluator. However, structural extraction and changes
to numerical policy must be separate review units. Temporarily retaining an old
cutoff while moving code does not make it a permanent requirement.

The linear-index API is a compatibility adapter, not the definition of a unit of
work. Python may change when needed for MPI orchestration. Existing CUDA code is
archived on `dev_cuda`; its design does not constrain this refactoring. Future
GPU optimization targets high-end devices such as GH200, not the local RTX A4000.
No CUDA implementation or cluster experiments belong to this initial refactoring.

## Review is a development requirement

Each code commit should answer one architectural question, have an observable
effect or a clear responsibility transfer, and leave the CPU evaluators usable.
The review unit includes the extraction, its immediate caller migration, and the
tests that establish its contract. Avoid speculative frameworks that acquire a
caller several commits later. Avoid unrelated formatting and renaming.

Use approximately the stages below, not a fixed line or commit count. Split a
stage when understanding it requires two independent explanations; combine tiny
supporting edits when they have no useful meaning separately. A correlator
migration may need its own commit because it introduces two-fragment joins.
Each unit must be understandable without reading the entire future architecture.

Pair each completed code commit immediately with a separate documentation commit
updating `work_log_backbone.md`. Use terse house-style subjects, for example:

```
[bs] extract self-energy paths
[docs] record self-energy path extraction
```

The work-log entry identifies the code hash and contains:

- The problem, illustrated with a small diagram or call flow where useful.
- Before and after: which function owned the responsibility, and which owns it now.
- Why this is the next useful step, including temporary compromises.
- A short reading order of the principal functions/files, with links.
- Behavior preserved and any intentional behavior change.
- Validation commands and outcomes, including limitations or outstanding failures.
- The next step, and deviations from the plan that affect later review.

Aim for a few paragraphs per entry, not a transcript of tools or debugging.
Maintain a compact stage/code-commit/status table at the top of the work log.
Record meaningful dead ends only when they explain the chosen design. Do not
mix documentation with code commits or silently rewrite completed review units.
Development can continue between these checkpoints without requiring user approval
after every commit. Unexpected architectural changes should be explained and the
plan revised before building further layers on them.

## Keeping the evaluators working

Retain the public CPU entry points during extraction. Initially the current
evaluator calls a newly extracted helper and then its existing arithmetic routine.
As preparation becomes explicit, pass the prepared description into that routine.
Replace responsibilities in place instead of maintaining two complete production
evaluators. The frozen baseline remains available through Git and, if useful, a
separate worktree/build for targeted comparisons.

Short-lived adapters may translate the old mutable `Backbone` into an owned
description. Each adapter must have a named removal stage. Descriptors must not
retain references to mutable backbone state that is reset for the next index.
Tests may briefly compare two paths through a narrow seam; remove redundant paths
after migration. Historical output is a diagnostic, not an independent oracle.
Keeping public functionality working does not mean promising indefinite source
compatibility for internal backbone methods.

## Existing C++ test dependencies

Inspection at baseline `2b0fc5c` found five test translation units that directly
construct `Backbone` or `CorrelatorBackbone`. The coupling is concentrated, but
extends beyond the two files named `backbone_*`. Other tests depend on the
flat-index contract without constructing a backbone. Test migration is part of
each responsibility transfer, not a final cleanup after changing all APIs.

| Test files (relative to `test/c++/`) | Current dependency | Migration responsibility |
| --- | --- | --- |
| `backbone_indexing.cpp` | Three tests exercise mutable setters/reset, getters, public `prefactor_sign`, three assignment encodings, kernel placement, invalid indices, counts beyond `int`, and overflow. Expected diagram data are derived from explicit rules in the test. | Stages 2–3: retain those expectations for shared preparation; keep encoding/count/error tests on the compatibility decoder. Replace reset-specific checks with assignment independence/lifetime checks when mutation disappears. |
| `backbone_parity.cpp` | Four tests exercise `get_parity`, special endpoint mutation, the constructor's `n_int` default, and external-statistics sentinels `-2`/`-3`. Includes a hand-derived parity table and rejection of unset indices. | Stage 3: preserve the parity table and mixed-statistics coverage against explicit interaction/observable statistics. Keep sentinel/default-constructor checks only while the legacy adapter exists; test missing assignments in the new representation. |
| `block_sparse/sparsity_invariance.cpp` | Constructs both backbone types and passes them into dense self-energy/paired routines and dense/block-sparse `eval_correlator`. Tests partition equivalence, orbital indexing, and a correlator with an empty second fragment. | Stages 1, 4–5: migrate calls to prepared contributions or suitable public APIs, preserving the numerical comparisons and deliberately unusual topology. Do not lose the fragment regression merely because the solver does not normally enumerate that topology. |
| `dense/correlator_statistics.cpp` | A helper constructs `CorrelatorBackbone`, reads evaluator dimensions/counts, and calls `eval_correlator` with explicitly forced statistics to distinguish correct from incorrect results. | Stages 3 and 5: retain this negative control using an explicit observable description or a narrow test helper. Do not reduce it to agreement between two calls that infer statistics identically. |
| `one_time_correlator_all_evals.cpp` | Its sign-sensitivity helper decodes indices with `CorrelatorBackbone`, reads orbital assignments, and resets it to select terms whose signs change under deliberately wrong statistics. | Stages 2–3 and 5: preserve independent sign-sensitivity checks while adapting assignment access. This is a direct backbone dependency inside an otherwise shared evaluator suite. |
| `flat_index_all_evals.cpp`, `flat_index_range.cpp`, `dense/dynint_parity.cpp`, and `test_utils/evaluators.hpp` | Indirect coupling to counts, orbital-fastest/direction-slowest enumeration, singleton/vector selections, paired-direction seeds/partners, and call independence. `flat_index_range.cpp` additionally requires `set_flat_index` in the error message. | Stages 2 and 7: preserve selection semantics through the compatibility layer without imposing its ordering on internal execution. Replace internal function-name assertions with relevant public error guarantees when validation moves; retain invalid-input, overflow, empty-selection, and recovery coverage. |

The shared evaluator adapters are useful insulation: many result comparisons can
remain unchanged while adapter calls migrate. They do not make the tests wholly
independent of shared preparation. `make_reference()` selects an evaluator from
the same registry, and the correlator sign-sensitivity helper itself uses
production decoding and topology functions. Preserve hand-derived indexing and
parity expectations and the existing analytic/closed-form tests; add independent
small references only where this inspection identifies a gap. Agreement between
two evaluators using the same new recipe is insufficient by itself.

For each affected code commit, the paired log entry must identify which tests
were adapted, what invariant each still checks, and any implementation-specific
assertion deliberately retired. Keep new preparation tests separate in purpose
from legacy enumeration-adapter tests: a regrouped or batched executor need not
expose one task per historical flat index. When retaining that API, selecting an
index must still select the same mathematical contribution, including paired
semantics. If the public selection contract changes later, document that as an
explicit API change with coverage of the replacement partitioning contract.

## Proposed sequence of review units

### 0. Establish the baseline and semantic checklist

Configure, build, and run the CPU suite on this branch. Reuse
`test/c++/test_utils/evaluators.hpp` and the shared flat-index, self-energy,
single-particle Green's-function, and correlator tests. Record existing failures
separately from changes. Add only missing contract tests justified by inspection.
Use the dependency inventory above to assign test migrations to their respective
code commits before changing the backbone interfaces.

Write down sign ownership across C++ and Python, forward/backward orientation,
reflection, conjugation, special endpoint sums, and input/output cutoff sites.
Choose a small common example to follow through subsequent log entries. Include
a partitioned and dense version of the same model and a retarded interaction case.
Acceptance: a reproducible baseline and an explicit list of numerical assumptions.

### 1. Make the ordinary self-energy path an explicit value

Extract block-path discovery from `find_path_self_energy` into a bounded visitor
or iterator producing owned path descriptions. Initially keep existing traversal,
symmetry grouping, masks, and arithmetic. Include the output sector, ordered
propagator sectors/dimensions, and the special-endpoint selection. Immediately
consume each description with the current evaluation routine.

This is a deliberately narrow intermediate boundary, not the final task format.
Acceptance: known paths and closures can be inspected without performing diagram
contractions; the existing numerical tests still pass. Dense code is unaffected.
The path description must record the terminal sector as well as the output
sector: the old walker does not explicitly check their equality. Do not silently
introduce a closure rejection in this extraction; first characterize it with
small path tests. A descriptor owns its sector/dimension arrays. The initial
stream can reuse one descriptor between callbacks, but consumers can copy it to
retain it. Memory usage must be independent of the total number of paths.
Decode outside the walker; the temporary caller still owns the mutable backbone
and invokes the old arithmetic. This limitation is removed in stages 2–4.

### 2. Separate topology and assignments from mutable backbone state

Extract shared topology/endpoint information and immutable assignment data from
the current backbone, retaining the flat-index decoder as an adapter. Cover pole
sign-dependent kernel placement and prefactors explicitly. Switch existing callers
through a small adapter first; do not simultaneously redesign sector traversal.

Acceptance: index round trips, both orientations, parity, and stable scalar-kernel
placement agree with their specified semantics. Descriptions survive preparation
of the next index. No sector IDs or storage/device choices enter the shared layer.
Use two review units if needed: first introduce owned assignment/topology values
with the existing decoder as a bridge; then extract decoding and kernel-placement
rules into preparation and make the old API delegate to them. Avoid two permanent
implementations of the diagram rules. Migrate the direct indexing tests with the
second unit, retaining their independently specified expectations.

### 3. Make interaction and vertex preparation explicit

Represent interaction families, statistics, endpoint operators, scalar coefficient
indices, and remaining sums explicitly. Describe barred operators as partial
contractions; materialization remains a representation/executor choice. Resolve
orientation, conjugation, reflection, and parity without mutation during arithmetic.
Use the dense CPU evaluator as the simplest first consumer, including paired
self-energy evaluation. Migrate block-sparse vertex selection to the same meanings
in a separate review unit if needed.

Acceptance: ordinary, paired, mixed fermionic/bosonic, and retarded cases remain
covered. Shared signs have one owner; representation-specific algebra is explicit
where existing conventions differ. Do not combine this with symmetry regrouping.
Split explicit statistics/sign preparation, dense consumer migration, and
block-sparse consumer migration when the combined diff obscures those contracts.
Remaining endpoint sums are represented explicitly, not expanded eagerly into
every orbital pair merely to make a descriptor immutable.

### 4. Finish separating self-energy planning and execution

Combine the shared recipe with block-specific path discovery into a prepared
self-energy contribution. Move remaining operator selection, special-index sum
planning, and destination decisions out of the arithmetic routine. The executor
receives resolved operands or stable handles with documented lifetimes.

Represent mathematical dependencies rather than prescribing a serial BLAS call
list. Start with a streaming CPU consumer; do not enumerate all contributions in
memory or introduce a general graph scheduler. Separate model, mesh, topology,
and propagator lifetimes so numerical data cannot survive an incompatible update.
Acceptance: execution needs no flat-index decoding, connectivity search, or
backbone mutation; changed propagators cannot reuse stale numerical data.

### 5. Apply the separation to observables

Extract the single-particle and general correlator boundary descriptions,
left/right fragments, compatible-sector joins, external operator statistics,
and trace destinations. Keep these differences explicit instead of forcing
correlators into a self-energy-only descriptor. Reuse shared diagram preparation
for dense evaluation without a block-path walker.

Acceptance: general bosonic/fermionic correlators, endpoint checks, reflection,
and single-particle functions pass the existing shared tests and relevant analytic
references. Numerical routines no longer discover fragment connectivity.

### 6. Correct numerical and support policy in separate commits

Now audit and revise magnitude-based input masks, per-contribution cutoffs, and
running-output masks. Do not merely remove the final `1e-16` comparison while
leaving earlier drops in place. Distinguish exact structural zeros from numerical
approximations; document any retained approximation and its error justification.

Treat symmetry-set generalization as a separate change from cutoff policy.
The current grouping is not fully optimal: computational groups can share a
schedule while individual operators have different exact support. General logical
operators may require multiple source/target components. Retain validation that
the chosen physical partition supports the assumed block-diagonal quantities.
Do not add epsilon terms to physical operators to obtain convenient groups.

Make two review units: numerical zero policy, then support/grouping semantics.
The latter must separate each operator's exact support from group-level candidate
paths; it need not find an optimal partition or implement a new sparse matrix
backend. Characterize any admitted cross-set coefficient remainder before deciding
whether to retain it explicitly or reject an unsupported model. Never silently
drop a coupling merely because it does not fit a convenient execution group.

Acceptance: independent small references, dense/partitioned agreement, scaled
small contributions, and cancellation-sensitive cases support the intended sum.
Choose tolerances from numerical accuracy evidence, not from reproducing legacy
threshold decisions. Audit reference routines for the same masking assumptions.

### 7. Consolidate the interfaces and remove migration scaffolding

Route linear-index/range selections through logical work generation; keep MPI
partitioning behavior explicit and test coverage/no-double-counting. Remove the
temporary backbone adapters and redundant mutable selection logic. Update Python
and generated bindings together only where required by the chosen interface.

Acceptance: full CPU C++/Python tests pass, the shared layer is independent of
block sparsity, and dense and block-sparse preparation can be consumed by future
executors. Record remaining limitations before undertaking performance changes.

## Validation and later optimization

Completion requires all seven implementation stages, including numerical-policy
validation and removal of temporary preparation adapters, not just a new set of
descriptor types. Existing self-energy, paired dense evaluation, single-particle
functions, and general correlators must remain usable from C++ and Python.
Finish with a short map from the old functions to their new owners, a full CPU
test result, and a list of deliberately deferred optimizations in the work log.
Performance optimization and a replacement CUDA executor are subsequent work.

Use focused semantic tests per code commit and the full CPU suite at the main
migration boundaries and completion. Include partition equivalence, nontrivial
complex values, both orientations, retarded interactions, and observable joins.
Do not require bitwise comparison when arithmetic changes; explain discrepancies
with independent accuracy evidence. No numerical-policy change may hide inside a
commit described as a mechanical extraction.

Keep targeted CPU timing checks separate from correctness evidence. Once the
interfaces settle, evaluate preparation overhead and bounded memory consumption
on a few representative cases before adding caching, reordered reductions,
cohorting, or parallel execution. These are subsequent review units with their own
motivation and measurements, not hidden additions to the refactoring. The old
CUDA experiments remain evidence for that later phase. Arrhenius has a shared
100 GPU-hour/month budget; this CPU phase requires none of it.

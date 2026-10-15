# Diagram/backbone naming work log

## Objective and scope

Complete and implement `plan_backbone_rename.md` on `dev_backbone`, preserving
CPU evaluator behavior and Python compatibility. The user added source
implementation to the persistent task on 2026-10-16. Use terse code commits,
with separate plan/work-log commits. No abstract hierarchies, GPU implementation,
cluster runs, history rewriting or pushes. Preserve unrelated worktree changes.

## 2026-10-16: plan completed

Inspected shared preparation, both evaluators, block paths, correlator discovery
and joining, kernel binding, numerical data, generated-wrapper selection and
keywords, tests, CMake discovery and the preparation benchmark caller.

Completed the plan with concrete records and explicit half-open vertex/edge
intervals. A correlator has a trailing edge to beta; self-energy needs sectors
outside the first/last operators. The sector-slot examples account for both.
Retained metadata owns its arrays; numerical views borrow live owners; correlator
reuse uses caller-local path/result indices. No mandatory CPU execution schedule
or host-view transport representation is imposed on a future GPU port.

Public Python names and keyword arguments are preserved, including `topology` at
the wrapped boundary; `topology_arr` is used internally. Prepared types/ArcData
are not exported. The plan contains six code stages, each followed by a separate
work-log commit, and an adversarial self-review of indexing, lifetimes, signs,
namespace shadowing, wrapping and future GPU constraints.

Validation at this checkpoint: source inspection and document consistency only.
No code or benchmark changes; no new numerical results claimed. Next: record the
CPU baseline, then implement stage 1. Existing unrelated state is `.gitignore`,
AGENTS files, CUDA smoke test, Python exp_order_conv and tools; leave it alone.

The goal tracker initially covers planning only. Its API cannot edit an active
objective; after completing that planning milestone, create the authorized
implementation goal immediately and continue without waiting for user input.

## Stage 1: shared descriptions (`c4d38f1`)

Moved shared topology, indices, diagram, arc/catalog, correlator statistics and
sign descriptions under `triqs_xca::diagram`. Consolidated the existing topology
utilities there and applied the agreed type/function names. FlatIndexSelection
remains at the root compatibility boundary. Updated in-tree C++ tests and the
benchmark caller directly; no permanent legacy aliases were added.

This stage deliberately retains the old member layouts (including flat identity
and kernel fields) so namespace/file movement can be reviewed separately from
extraction. Fully qualified shared names make the upcoming block_sparse::diagram
namespace unambiguous; local naming cleanup remains stage 6.

Validation: initial CPU build and all 73 tests passed (59.08 s). Reconfigured and
built after the move; six focused topology/indexing/parity/contraction/prepared
execution tests passed. Python binding regeneration introduced no tracked changes.
Logs: `/tmp/xca-rename-baseline-{build,test}.log` and
`/tmp/xca-rename-step1-{configure,build,test}.log`.

Next: extract ArcData/TauKernelValues, explicit TauKernelPlacement and the bind
namespace, preserving discrete signs and barred-operator pole selection.

## Stage 2: arc data and kernels (`fc511be`)

ArcData now owns the numerical coefficients/values and TauKernelValues. The
physical coefficient helpers remain in hyb. Diagram owns TauKernelPlacement;
operator pole slots remain on Vertex, independent of vertex scalar factors.
Discrete direction signs remain separate from kernel prefactors. Binding is
`triqs_xca::bind::tau_kernels`; arithmetic helpers live outside bind.

Removed stored flat identity from Diagram and the old path record. Flat decoding,
validation and 64-bit counts remain at the boundary. Tests retain their explicit
expected digits/factors and independently owned arrays; only storage-identity
assertions were removed. The operator/kernel pole-slot test now checks each
role separately, including barred operators with no vertex scalar factor.

Validation: reconfigure/full CPU build passed; ten focused tests passed (3.61 s),
covering kernel views/reflection, indices/signs, contractions, prepared evaluators,
retarded interactions and all-evaluator correlators. Generated Python wrappers
remain unchanged. Logs: `/tmp/xca-rename-step2-{configure,build,test}.log`.
Next: compact half-open backbone intervals and SelfEnergyDiagram metadata.

## Stage 3: backbone intervals (`c1522aa`)

Added block_sparse::diagram::BackbonePath with explicit half-open vertex/edge
bounds and compact owned sector/dimension arrays. Global sector-slot accessors
translate directly to local storage. SelfEnergyDiagram owns one path and keeps
endpoint symmetry sets/output destination outside it. Correlator paths use the
same type; the segment role is an explicit shared CorrelatorSegment argument.

The path includes outer self-energy boundary sectors and the correlator's
trailing edge to beta. Existing mask checks and candidate/exact support ordering
remain unchanged. Tests retain their hand-written sector/dimension expectations,
with extra assertions for compact size, full-path boundaries and single-edge
segments. Numerical helpers now read dimensions through the path.

Temporary adaptation: the old numerical operand containers remain, including
full-sized correlator view vectors and the self-energy aggregate's `path` field
holding SelfEnergyDiagram. Stage 4 replaces those containers with BackboneOperands;
this commit isolates the logical interval change from numerical-view extraction.

Validation: full CPU build passed; six focused path/mask/ownership/numerical and
all-evaluator correlator tests passed (3.60 s). Logs:
`/tmp/xca-rename-step3-{configure,build,test}.log`.

## Stage 4: backbone operands (`73a58a7`)

Both full self-energy paths and correlator segments now bind through
block_sparse::bind::backbone into one BackboneOperands type. Propagator views
and optional vertex views cover only their intervals; absent vertices explicitly
mean completion operands supply them. Direct accessors translate diagram indices.

SelfEnergyOperands composes BackboneOperands and EndpointSumOperands, with output
and sign bookkeeping. bind::endpoint_sum and bind::self_energy separate numerical
selection from execution; exact nonclosing-path policy remains before arithmetic.
Vertex binding moved under bind. The old padded fragment operand structure and
self-energy contribution structure were removed, without compatibility aliases.

Validation: CPU build passed; six focused evaluator/partition/correlator tests
passed (5.30 s). Added checks of compact view counts, missing special endpoints,
and the single-edge tau-to-beta propagator. Retained numerical-view ownership
and hand-written shape/reference comparisons remain. Logs:
`/tmp/xca-rename-step4-{configure,build,focused-build,test}.log`.
Next: separate correlator completion metadata from its bound external operators.

## Stage 5: correlator preparation (`b58aa6b`)

Replaced the mixed correlator_plan file with diagram/correlator descriptions and
bind/correlator numerical selection. CorrelatorDiagram contains caller-local
path-table indices and external-operator choices. CorrelatorOperands replaces
the old join structure: it binds external matrices only; computed segment results
remain separate inputs. No extra enclosing aggregate or class hierarchy.

The evaluator retains compact paths/results for one shared Diagram, evaluates
each segment once, and streams compatible completions in the existing order.
The preparation visitor reads connectivity/masks but returns no numerical views.
Binding needs the prepared record, retained paths and operator owners; it does
not require an otherwise unused shared Diagram argument.

Validation: full CPU build passed; four focused metadata/binding, prepared and
all-evaluator correlator tests passed (3.61 s). The synthetic completion test now
checks that filtering a path table changes table indices without confusing them
with sector IDs, and verifies bound operator addresses independently. Logs:
`/tmp/xca-rename-step5-{configure,build,focused-build,test}.log`.
Next: final naming cleanup, full CPU/MPI validation and benchmark compile check.

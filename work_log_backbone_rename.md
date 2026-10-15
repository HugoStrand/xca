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

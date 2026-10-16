# Shared and block-sparse backbone boundaries

Follow-up to the adversarial review of plan_backbone_rename.md, authorized
2026-10-16 on dev_backbone. Keep code and documentation commits separate.

The review found shared geometry inside block-specific paths, an enum mixing
correlator geometry with CPU scheduling, and block record headers importing host
preparation dependencies. The correction has three stages:

1. Share backbone ranges between dense and block-sparse evaluation.
2. Give Diagram a shared Backbone owner, separate whole-diagram factors, and
   represent the final correlator propagator edge explicitly.
3. Use ranges for block discovery, separate CPU forward/backward execution, and
   isolate plain block records from host discovery declarations.

No class hierarchy, generic scheduler, GPU implementation or timing experiment.
Preserve arithmetic order, public Python compatibility and unrelated worktree
changes. Future GPU targets remain high-end devices such as GH200.

## Stage 1 — shared geometry (`a98491c`)

Added diagram::BackboneRange with explicit half-open vertex/edge bounds and
validation of an alternating chain. Block-sparse BackbonePath now composes the
range and adds only sector metadata. Both dense evaluation and block discovery
obtain the two correlator ranges from the same common helper. A range may contain
one edge and no vertices; it may include either outer operator. Invalid gaps
between the vertex and edge intervals are rejected.

The CorrelatorSegment enum temporarily remains as an execution selector; stage 3
removes it after the backbone owner and full edge extent are explicit.

Validation: configured and built the CPU tree; six focused shared geometry,
path ownership, prepared evaluator and all-evaluator correlator tests passed
(3.71 s). Generated wrappers are unchanged. Logs:
`/tmp/xca-boundaries-step1-{configure,build,test}.log`.

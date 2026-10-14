# Full-shape buffer lifetime experiment

The target is GH200-class hardware; local RTX A4000 runs screen correctness and
mechanisms. Preserve production, the row experiment and all other candidates.

`LifetimeDiagramEvaluator` is generated from current production with checked source
seams. Each SE/correlator edge drains and releases previous DLR scratch, converts the
weighted source, drains and releases its value tensor, converts T, drains and releases
T's value tensor, then allocates the same output shape and calls the unchanged full
convolution. Both coefficient inputs remain live through completion. Scratch is released
before the next vertex/special phase. GEMM dimensions, operand order, solves, cutoff
and path order are unchanged. Public memory estimates intentionally retain production's
conservative bounds pending phase-by-phase admission audit.

Build `block_sparse_lifetime_tests`, `block_sparse_lifetime_benchmark` and
`block_sparse_lifetime_cutoff_audit`. The experiment is omitted in CPU-only builds.
Tests inherit all current production C++ evaluator controls and add exact baseline
comparisons for changed input/masks, both output placements, repeated calls, budget
rejection and recovery. The separate cutoff diagnostic bisects an active physical OCA
call to adjacent representable scales at two ranks and checks nine neighbors each.
Its completed process alone does not prove acceptance: inspect every decision and
matrix error. The extra rows column is zero (no chunking) to keep the prior audit CSV
format comparable.

The complete-call benchmark reuses block_sparse_common's bounded fixture, CPU screening,
exact self-replay, warm timings and one selected topology/index. Both reference and
candidate use library sparse products and reference special vertices. Report synchronized
latency including allocations/drains, explicit bounds separately from sampled global
free-memory deltas, and changed-G/host-output qualification. Additional synchronizations
and allocation churn can erase a memory benefit. No dispatch/default, new API or target
hardware qualification follows from this experiment alone.

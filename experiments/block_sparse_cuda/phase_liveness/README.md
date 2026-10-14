# Full-shape buffer lifetime experiment

The target is GH200-class hardware; local RTX A4000 runs screen correctness and
mechanisms. Preserve production, the row experiment and all other candidates.

`LifetimeDiagramEvaluator` is generated from current production with checked source
seams. Each SE/correlator edge drains and releases previous DLR scratch, converts the
weighted source, drains and releases its value tensor, converts T, drains and releases
T's value tensor, then allocates the same output shape and calls the unchanged full
convolution. Both coefficient inputs remain live through completion. Scratch is released
before the next vertex/special phase. GEMM dimensions, operand order, solves, cutoff
and path order are unchanged. The experiment bounds temporary phases by seven maximum rank tensors including
convolution scratch (rather than seven caller plus five DLR tensors). Solves require
at most four, special fallback six, and vertices/reflection two. Correlator retained
fragments, input/output, endpoint/result and sparse/mapped reserves stay unchanged.
NCA bounds are unchanged. Admission drains stale scratch from earlier failures before
new caller storage is reserved, after checking the explicit budget. This estimate
excludes opaque allocations, which still require the existing free-memory headroom.

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

The phase bound uses S for a maximum `r*D*D` complex tensor:

| Phase | Concurrent temporary payload (at most) |
| --- | --- |
| First coefficient solve | T, weighted values, source coefficients, RHS: 4S |
| Second coefficient solve | T, source coefficients, T coefficients, RHS: 4S |
| Full disjoint convolution | two coefficient inputs, output, four DLR tensors: 7S |
| Special reference vertex | T, output, kappa, weighted kappa, mu, term: 6S |
| Mapped special vertex | four rank tensors plus existing mapped metadata reserve |
| Ordinary vertex / reflection | two rank tensors |
| Correlator endpoint | applied, product, square: 3S, plus existing result/trace reserves |

For correlators, retained right/left fragments use their existing two-function reserve.
Each fragment's block mapping composes the model's injective connections; for a partial
permutation p, sum(d_b*d_p(b)) <= sum(d_b^2). Thus rectangular fragments do not require
padding every slot to D. Inputs, complete outputs and all old non-scratch reserves are
still counted. Stale scratch is drained at admitted entry before any new caller arrays.
This is an explicit-payload argument, not a bound for opaque library allocations or
host atom construction. No full fourteen-mode physical accuracy/capacity claim follows.

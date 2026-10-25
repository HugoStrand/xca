# Diagram refactoring work log

## Starting checkpoint — 2026-10-25

Branch: `dev_backbone`, code baseline `ad9e470`.
Plan: [plan_diagram_refactor.md](plan_diagram_refactor.md).

The user approved arc objects, integrated backbone kernel recipes, a single arc
association per vertex, explicit endpoint operations and the recommendations
recorded in the plan. Implement these as separate reviewable code checkpoints,
with documentation commits following each checkpoint.

Baseline from the preceding change: all 76 CPU/Python CTest entries passed in
56.94 s; backbone_cpu compiles; generated Python bindings were unchanged.

Unrelated initial state to preserve: modified .gitignore; untracked AGENTS.md,
AGENTS.md.bak-20261009, test/c++/test_nda_cuda_smoke.cpp,
test/python/__pycache__/, test/python/exp_order_conv/ and tools/.

Status: plan recorded; implementation pending.

## Checkpoint 1 — integrated backbone kernels

Code: `0f9c8fe` — `[diagram] integrate backbone kernels`.

Moved kernel normalization into Backbone, attached each vertex kernel recipe to
its vertex and replaced dense edge-sign rows with ordered KernelFactor lists on
Edge. prepare_backbone now returns the coherent operator/kernel description;
prepare_vertices remains callable, with its implementation moved into backbone.cpp.
Removed TauKernelPlacement and the old tau_kernel.cpp. The kernel recipe header
remains for the small factor/prefactor types.

Both evaluators and the benchmark use the new ownership. Tests retain their
independent expected signs and numerical values, with access-path changes and
explicit list inspection. The edge-order tests still cover repeated poles and
avoidable overflow/underflow. Existing operator pole slots and nested vertex
kernel pole slots are transitional until checkpoint 2.

Review caught an accidental recursive call during extraction. It was fixed while
a compiler job was already running; the stale compiled object caused the first
suite to fail. Forced recompilation of backbone.cpp resolved that failure. No
failing state was committed. Subsequent build and all 76 CPU/Python CTest entries
passed (57.67 s); backbone_cpu builds; generated Python bindings are unchanged.
Tests used OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, parallelism 2.

Logs: /tmp/xca-diagram-step1-corrected-build.log,
/tmp/xca-diagram-step1-corrected-test.log and
/tmp/xca-diagram-step1-benchmark.log. git diff --check passed.

Status: checkpoint 1 complete; arc-object migration next.

## Checkpoint 2 — unified arc indexing

Code: `6e84f1a` — `[diagram] store arc choices together`.

Arcs is now a vector of Arc objects, including an explicit special entry at zero.
Direction, retained_flavour and pole_index live together. Both vertices of an arc
refer to its topology row through arc_idx; duplicated vertex flavours and both
operator/kernel pole slots are gone. Numerical kernel and operator helpers resolve
choices through that arc. Prefactor arrays use the same arc indexing, with an
identity entry for the special arc. The decoder retains its original digit order.

Updated dense and sparse consumers, independent indexing/sign fixtures and the
benchmark. Tests check special metadata, both vertex references and copy ownership.
The correlator sign oracle still uses its independent parity computation; only
its lookup of a vertex's flavour changed. No numerical conventions were changed.

Validation: CPU build and all 76 CPU/Python CTest entries passed (57.13 s);
backbone_cpu compiles; generated Python bindings unchanged; git diff --check passed.
Logs: /tmp/xca-diagram-step2-{build,test,benchmark}.log.

Status: checkpoint 2 complete; explicit endpoint operations next.

## Checkpoint 3 — explicit endpoint operations

Code: `91d5a68` — `[diagram] expose endpoint operations`.

Diagram retains Kind and exposes endpoint_ranges() and self_energy_endpoint_sum().
The named before_special/after_special ranges exclude the two distinguished
vertices. Self-energy can have an empty tail; a correlator always retains its
trailing propagator. BackboneRange now distinguishes a fully empty range from a
single-edge range and rejects vertex-only intervals.

Dense evaluation uses three small private traversals: initialize the first
segment, extend an existing result through the self-energy tail, and evaluate the
reflected correlator segment. Endpoint summation remains its own operation.
Paired evaluation still shares the initial segment and subtracts the opposite
orientation before traversing the tail. Normalization is applied once, in its
previous arithmetic order. Sparse correlator evaluation uses the same named
geometry without changing its path-discovery algorithm.

Prepared numerical entry points reject the wrong kind before changing buffers.
The sparse self-energy path visitor also checks kind. Empty sparse ranges emit no
independent paths, and numerical segment routines reject empty operands before
accessing an edge. These choices distinguish extending an existing result through
an empty tail from inventing an independent contribution for that empty range.

Added tests for first-order and higher-order final endpoints, a nonempty tail,
both special orientations, correlator endpoint-sum rejection, empty sparse walks,
wrong-kind rejection and sparse evaluator recovery. The first build exposed a
new test accessing private Sigma storage; changed it to check public scratch
buffers and recovery through the public API, without exposing private state.

Validation: CPU build and all 76 CPU/Python CTest entries passed (57.16 s), including
the new cases within existing executables; backbone_cpu compiles; regenerated
Python bindings unchanged; git diff --check passed. Logs:
/tmp/xca-diagram-step3-{build,test,benchmark}.log.

## Final review and completion

Reviewed the complete diff against ad9e470, with particular attention to:

- Flat-index digit order and paired-range ownership: unchanged, covered by the
  independent indexing fixtures and evaluator comparisons.
- Kernel order, reflection and normalization: edge lists preserve increasing arc
  order and repeated pole references; the existing overflow/underflow cases pass.
  Moving the prefactor does not duplicate it across correlator segments.
- Boundary geometry: first-order cases, final special vertices, trailing
  correlator edges, and empty self-energy tails are explicitly distinguished.
- Owning copies: arc choices and backbone recipes remain independent values.
  Mutating a prepared value can invalidate its recipes; re-preparation is the
  documented contract, with no hidden invalidation mechanism.
- Separation: shared diagram types contain no sparse-sector state or execution
  workspace. The changes do not require GPU executors to receive every CPU path.
- Numerical/Python compatibility: existing tolerances and public Python
  signatures were retained. This validates the CPU implementations only.

No further correctness issue was found in this review. Known tradeoffs remain:
per-edge vectors allocate separately; prepared descriptions are immutable by
convention; the low-level topology/correlator geometry helpers remain available
for existing callers. No performance claim or GPU validation is made. Future GPU
optimization remains aimed at high-end devices such as GH200.

Status: all planned implementation and review checkpoints complete. Code and
work-log commits are separate. Unrelated initial worktree changes were preserved;
no commits were pushed.

# Small-matrix CUDA benchmark and optimization work log

## Current checkpoint — 2026-10-11

Preparation goal is COMPLETE. The separate optimization-exploration goal is ACTIVE.
Subsequent thread turns are verified as gpt-6.1-sol / medium / standard. The current
turn performs only handoff bookkeeping; optimization implementation starts on the
next turn with E1 serial_stream ordinary-SE. Then evaluate all priority strategies
per plan, using separate folders and preserving the frozen baseline.

Custom quota recovery is active in cbuild-supervisor-small/exploration/; exact new
goal binding and thread_active were verified. Boot registration points to this config.
Preparation watcher was stopped; the shared Codex server was not restarted. No live
benchmark/build/profiler job remains. Do not recreate the exploration goal.

Branch dev_cuda. Production CPU/CUDA evaluators at baseline commit064b64e remain
unchanged. Keep experiments in individual folders under experiments/dense_cuda_small/;
never replace the production baseline. Commit code separately from plan/log, no push.
Preserve modified .gitignore and untracked AGENTS.md, AGENTS.md.bak-20261009,
test/c++/test_nda_cuda_smoke.cpp and test/python/exp_order_conv/.

## Preparation progress (supersedes initial numerical investigation below)

Benchmark contract v2 is committed in9bedfa9/c1a87d7. Both CPU/CUDA builds and three
benchmark-contract tests pass; all bounded se/pairs/spgf smoke cases through order12
pass against the unchanged CUDA baseline. CPU discrepancies at high order are
reported separately, low-order CPU agreement remains required, and the relative
parity gate rejects zero even for1e-41 contributions. Changed-input/duplicate/empty
cases are validated. Fresh timed evaluators charge setup and first-call lazy caches.

Final serial measurement batch is native session97265: quick01/02/03, throughput,
coverage, rank26. A queued CPU-prefix command omitted --backend cpu and is expected
to fail after those six GPU runs; preserve the failed output under cbuild-small-runs/
and rerun the CPU control with the flag, then run the planned N4 order6 profile.
Do not restart any live benchmark or run GPU work concurrently. Old preliminary
prewarm-contract data was moved intact to cbuild-small-runs/prewarm-contract-baseline.

Draft plan_dense_cuda_small.md exists with detailed E1–E4 strategies, reviewed
stream/status/pointer-table hazards, shared-LU batching layout, and success/stop rules.
Still needs measured baseline/profile findings and final self-review before model
handoff. No optimization implementation has begun.

## Initial investigation (historical)

New shared harness `benchmarks/dense_cuda_small/harness.hpp`, executable
`examples/dense_cuda_small_benchmark.cpp`, runner/CSV comparison scripts are written.
Both CPU/CUDA builds compile. Options bound N to2/4/8/16, orders1–12, fixed crossing
and chain topologies, explicit sampled-index counts, methods and repeated timings.
Raw CSV includes exact selected indices, result norms/errors, setup and per-index
costs. Runner saves source/binary hashes, prefixes/libraries and raw timing samples.
The production evaluators are untouched.

CPU order1/6/12 smoke passed. GPU order1/3/8 smoke exposed a high-order comparison
outside the initial tight relative gate (1e-10 plus1e-28 absolute). Investigating
precise error before deciding acceptance; do not weaken it blindly. Some order8
contributions are ~1e-23 and relative errors are large despite tiny absolute error.
Diagnostic job99949 rebuilds/runs order8 with scientific error reporting. Check its
status and /tmp/xca-small-diagnostic.csv before restarting. Previous build jobs ended;
preparation build/smoke logs /tmp/xca-small-{cpu,cuda}-build.log and related CSVs.

Supervisor now accepts configured effort (default remains medium); 24 tests passed,
including exact model/effort preservation. Preparation runtime:
cbuild-supervisor-small/preparation/. Existing boot marker now launches its config;
old completed-goal runtime preserved. Goal identity/pause/quota checks unchanged.
At transition, use a fresh exploration runtime/objective, select Sol6.1/medium using
`thread/settings/update` (installed protocol supports it), and verify thread/read.
Do not launch a second inference worker or change permissions. No model change yet.

Next: resolve benchmark numerical coverage; validate runner/compare and CPU/GPU
high-order/method cases, record quick/throughput baselines and a small N4 profile.
Then create plan_dense_cuda_small.md with strategy priorities, isolated experiment
layout, robust numerical gates, measured success/stop criteria and handoff. Complete
preparation goal only after its artifacts are committed; create exploration goal and
hand off to the requested model. Read this checkpoint, AGENTS.md and status on resume.


## Final preparation acceptance — 2026-10-11

Benchmark contract v2: fresh timed evaluator per case after correctness checks, setup
and lazy first-call costs visible, exact index selection preserved, raw samples and
provenance recorded. Seven successful baseline runs total224 backend/case groups.
Three quick repeats, throughput counts1/8/32/128, orders through12/method coverage,
rank26 control and CPU-only-prefix control are committed. Initial v1 runs and failed
CPU invocation (missing --backend cpu) are preserved under ignored cbuild-small-runs/;
corrected CPU control passed. Final measurements are under
benchmarks/dense_cuda_small/baseline/ and interpreted in RESULTS.md.

N4/r16 crossing B1: order6 CUDA4.43ms vsCPU.75ms; order8 CUDA6.27ms vsCPU1.03ms.
Throughput currently scales essentially linearly with index count. N4 order6 trace:
257 kernels and about335 device synchronizations per complete call. Three timed
traversals and123 contained phase ranges have no allocations/deallocations or explicit
host/device copies. Complete API has two malloc/free calls outside traversal; their
ownership is an experimental investigation target. Trace peak includes reference and
candidate/library allocations. Raw report/SQLite ignored in cbuild-small-profile/.

Both CPU/CUDA builds and all three benchmark-contract tests pass; actual GPU
per-index/aggregate checks, changed input, empty and duplicate indices pass through
order12 se/pairs/spgf. Five malformed CLI cases reject; mismatched comparison case sets
reject; self-comparison returns1x. Supervisor24 tests pass with phase-specific effort.
No production evaluator source changed:23 protected hashes match baseline064b64e,
also checked against every run's manifest. The previous production analytic suites
remain required for experimental changes; no production code change warranted
repeating all of them merely for this benchmark-only milestone.

Numerical limitation is explicit: high-order CUDA-reference parity is the experiment
gate (relative1e-10 with1e-280 floor); low-order CPU agreement also required. CPU
high-order discrepancy is reported separately. One order10 lane differs1.35 relatively,
while all individual absolute differences in that case are <=3.54e-34. Tiny cancelling
contributions need further accuracy investigation before scientific interpretation;
no gate was loosened to bless an optimization, and zero substitution is tested to fail.

Plan self-review used the measured profile and source/API constraints. Revisions cover
fresh-object timing, exact input/index parity, status/pointer-table asynchronous
hazards, graph invalidation/setup amortization, wide shared-LU RHS batching, tails,
deterministic reduction and bounded memory. Priority experiments:serial_stream,
graphs,index_batch,fused_small; CPU hoisting is an optional control. Each gets its own
folder/namespace/target/results, baseline source remains untouched. Successful and
negative measurements count; speculative rejection alone does not. See the plan's
explicit success and completion rules. No exploration code was written during
preparation. No subagents were used.

Implementation/benchmark commits:6095840 (effort-aware supervisor),9bedfa9 (harness),
c1a87d7 (fresh timing/profile boundaries),889dc11 (isolation rules), plus final data
commit. Plan and this work log are separately committed. No push; unrelated user work
is preserved. Native jobs97265/83256 and all build/test jobs are terminal; no live
benchmark remains. Read current checkpoint and AGENTS.md before continuing.


## Handoff completed

Preparation evidence commit6add3f7, plan-only97ed376, preparation checkpoint5e06d91.
Preparation goal completed after159,364 recorded tokens and1,526s (about25min).
`thread/settings/update` selected gpt-6.1-sol/medium/default; thread/read verified
model and reasoningEffort. Separate exploration goal created and active; current
server goal and ignored exploration config contain its full exact objective.

Immediate next action: read the plan, verify protected baseline hashes and current
status, create experiments/dense_cuda_small/serial_stream/ with a bounded ordinary-SE
prototype and independent build target. Begin with owned stream/handles and safe
status/pointer-table lifetime. Do not infer that deleting synchronization calls alone
is safe. Implement/measure the other E1–E4 priorities and useful combinations under
the new goal; report supported methods and negative results honestly. Model handoff
is now configured, so no further model-switch or preparation work is required.

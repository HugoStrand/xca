# Small-matrix CUDA benchmark and optimization work log

## Current checkpoint — 2026-10-11

Preparation goal is COMPLETE. The separate optimization-exploration goal is ACTIVE.
Subsequent thread turns are configured as gpt-6.1-sol / medium / standard. E1
serial_stream ordinary-SE is implemented, verified, measured and committed74f2adf.
E2 graph replay is implemented, verified, measured and committedc98801b. Next
implement E3 cross-index batching and E4 fused kernels, preserving each experiment
and the frozen production baseline. The exploration goal remains ACTIVE.

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

## Supervisor missed restart: 2026-10-11

User requested diagnosis of a missed restart after the expected 22:00 quota reset.
The detached watcher was alive, but its action stayed `waiting_thread_error` from
09:11:41 UTC until the user manually resumed the session. This branch returned
before querying quota permission. The installed protocol defines `systemError` as
a runtime thread state; only that error state could reach this branch among the
protocol's four states. The latest preceding turn failed at 09:11:35 UTC with
`usageLimitExceeded` and a workspace-credit message. During diagnosis the fresh
backend snapshot reported `ordinaryUsageAllowed: true`, primary usage 3%, secondary
16%, and no spend-control block. No historical quota snapshots exist to establish
the exact reset time or distinguish the backend's earlier credit message from a
window limit. A reset clock alone is never treated as permission to resume.

Fixed both the initial runtime gate and final pre-start checks: a loaded
`systemError` thread may recover only from its latest terminal quota failure, with
fresh explicit usage permission, the same failed turn rechecked, and existing
goal/branch/pause/grace/journaling guards preserved. Added a final quota-permission
recheck and persisted runtime/latest-turn diagnostic fields. The earlier reset
tests incorrectly modeled failed threads as `idle`; the transport regression now
keeps `systemError` until the new turn starts. Four added regression tests cover
both active/usageLimited goals, unrelated error states, a changed failure racing
recovery, and revoked permission during final checks. All 28 supervisor tests pass.

Stopped and relaunched only the dedicated tmux watcher; shared app-server and
active session were preserved. Live dry-run and relaunched process report
`thread_active`, with no pending start. Lock PID194125 and the existing @reboot
registration target the exploration config. This validates deployment and active
session isolation; automatic resumption after the next natural exhaustion still
needs production observation. Supervisor code/tests/docs are committed separately
from this checkpoint. No CUDA evaluator changes, no push, unrelated work preserved.

Exploration goal remains active. Resume the bounded ordinary-SE serial_stream
prototype next, following the plan and frozen baseline constraints above.

## E1 owned stream accepted — 2026-10-11

User requested persistent continuation. Fresh backend usage permission allowed
restoring the exploration goal from usageLimited to active (automatic provenance).
Do not recreate or complete this goal. Supervisor remains bound to it.

Committed74f2adf: isolated serial_stream namespace/standalone build, copied baseline
arithmetic with direct owned-stream cuBLAS/cuSOLVER adapters, immutable constructor
vertex pointer tables,65536 distinct persistent managed solver status slots, one
return-boundary stream fence and no shared NDA setting changes. Factory supports
ordinary SE only; unsupported methods reject. Headers retain unused copied API
declarations, but no experiment support is claimed for them. Calls are serialized.

Five isolated CTest entries pass: reused one-/two-fermion analytic and complex
rotation tests (experimental ordinary SE; baseline fallback for other methods),
reused DLR primitive tests and new API regressions. Invalid-prefix recovery,
>INT_MAX indices and owned-solver independence from NDA's shared synchronization
setting pass. Memcheck API tests report zero errors. All23 protected baseline
source hashes match064b64e; no production build/source change was required.

Seven serial benchmark runs contain210 backend/case groups: three quick repeats,
throughput B1/8/32/128, order12 coverage, rank26 and second seed31415. Per-index,
aggregate, changed-input, duplicate and empty parity are exact against frozen CUDA;
low-order CPU checks pass, high-order CPU diagnostics remain recorded separately.
Raw data/manifests/source hashes/comparisons are in serial_stream/results/.

N4/r16 crossing B1: order6 medians3.091/3.065/3.437ms vsbaseline4.407ms
(1.28–1.44x); order8 4.241/4.255/4.277ms vs6.236ms (1.46–1.47x).
B128 order6 362.9ms vsCPU92.44ms, order8 511.0ms vsCPU128.0ms. Throughput
remains nearly linear and about4x slower than CPU. E1 helps but is insufficient.

Trace N4/order6:257 kernels per index, zero traversal allocations/frees or explicit
host/device copies; one owned-stream fence. Complete API retains two malloc/free
calls and the benchmark's two device fences. Managed status migration131072bytes
per call is reported explicitly. Owned bytes370496; observed peak51,459,509bytes
includes reference/library allocations, not physical candidate residency. Async
phase ranges are host submission only; traversal encloses actual completion.
Raw Nsight artifacts are ignored under cbuild-small-profile/stream/.

No live GPU/build/test job remains. Code/results are committed separately from this
log. Next: E2 graph capture feasibility and replay with charged first-call cost,
content-keyed topology/index cache, changed G uploaded on every call, strict parity
and cache miss/rebuild tests; then implement E3 even if graphs help.

## E2 graph replay accepted — 2026-10-11

Committedc98801b: separate graphs folder/namespace, copied owned solver/evaluator,
retained E1 DeviceOps, one content-keyed cached graph/executable, ordinary indexed SE
limited to128 indices per call. Current G upload and output zeroing happen on every
call outside capture. First/cache-miss capture/instantiation is charged inside the
API timing and marked xca.graph.build; replay plus completion is inside traversal.
No production or E1 implementation change. E1 results additionally document setup
amortization. The runner hashes the parent experiment tree to include E1 dependencies.

Four isolated CTest entries pass: existing one-/two-fermion and complex-rotation
suites via an indexed graph adapter, and graph API regressions. Ordinary SE uses
graphs; unsupported methods remain baseline tests. API checks cover same-address
index/topology changes, cache hit/miss counts, empty vectors without replay,
>INT_MAX indices, invalid-prefix recovery, capacity rejection and shared-NDA-setting
independence. Memcheck API tests report zero errors. An initial test wrongly used
move assignment while asserting stable topology storage; changing assignment through
a view made the intended same-address test valid. No parity failure occurred.

Seven runs/210 backend-case groups pass strict exact CUDA-reference parity, including
high orders through12, rank26 and second seed. Three graph B1 medians: order6
2.859/3.132/2.833ms (1.41–1.56x frozen CUDA), order8 3.949/3.923/4.017ms
(1.55–1.59x). Order8 gains6.5–8.5% over E1; order6 is more variable and one repeat
regresses against its E1 counterpart. B128 warm338.2/464.0ms (orders6/8), first
529.3/739.3ms; against E1 measured first/setup penalty amortizes after about8/7
total calls. Still far slower than CPU; fixed-index reuse matters, not merely graphs.

Initial Nsight default graph-level recording hid replay node kernels/copies. Kept
that explicitly incomplete CSV/raw trace, then repeated with --cuda-graph-trace=node.
Auditor now rejects traversals with no visible kernels; rejection was checked against
the incomplete trace, and E1/node-level graphs re-audited successfully. Node-level
graph traversal:257 kernels,53248D2Dbytes, one graph launch/stream fence, no
allocations/frees or explicit host/device copies. Managed status131072bytes migrate
per call. Owned known arrays370496bytes exclude opaque graph/library allocations;
observed peak51,459,509bytes includes reference/library caches, not physical residency.
Raw node-level reports stay ignored in cbuild-small-profile/graphs/; data/reports are
tracked under graphs/results/. No live GPU/build/test job remains.

Next E3 implementation: use (r,B,N,N) storage, r*B tiny-matrix products and shared
r-by-r cached LU with B*N*N RHS columns. Operator/kernel/sign choices vary by lane,
so prepare immutable pointer tables and per-node row weights from lane Backbones
outside traversal; never overwrite queued metadata. Fixed-capacity chunk buffers
must mask inactive tails. Initially preserve sequential accumulation exactly by
gathering each finished lane into contiguous persistent device scratch (pitched D2D
copy), then the unchanged complex cuBLAS axpy in caller order. This avoids prematurely
introducing a reduction kernel or CPU summation reassociation; its overhead must be
measured. Wide GEMM/solve algorithms may alter rounding, so investigate any strict
parity failure without weakening gates. Then E4 fused N4 and useful combinations.

Code/results and checkpoint remain separately committed; no push. All23 baseline
hashes match064b64e. User work preserved; goal remains active under the supervisor.

## E3 bounded prototype and upload handoff checkpoint — 2026-10-11

Committed3d7c826. Isolated index_batch implements ordinary SE with physical(r,B,N,N)
storage, shared cached r-by-r LU with B*N*N RHS, wide rank transforms, lane-specific
immutable metadata and content-keyed single-chunk schedule reuse. Capacity1..128,
long indices, inactive tails, deterministic sequential device accumulation; production
23 baseline hashes unchanged. First/cache-miss metadata cost remains charged to API.

Five E3 CTest entries pass (analytic adapters, rotations, wide primitives and API).
Capacities1/8/32/128, different per-lane data, r16/r26, duplicate/empty/changed inputs,
cache invalidation, B+1/2B+3 tails and >INT_MAX are checked without weaker tolerances.
Initial uncached probes through order12 matched unchanged CUDA exactly. Primary
source-hashed cached timings, memory traces and sanitizer evidence remain pending.

A wide primitive exposed an upload race absent from the original fixture checks:
pageable default-stream cudaMemcpy may return before DMA completion, while our owned
nonblocking stream is independent. r16/B128 reconstruction error was0.14100087395.
A diagnostic default-stream fence removed the error. All three experiments now use
an explicit persistent CUDA event handoff at external/default-stream upload boundaries,
without introducing per-GEMM synchronization. No arithmetic/tolerance change. Delayed
producer regression fails when its handoff is deliberately omitted, then both primitive
tests pass with restored source. Negative/positive logs plus all three serial CTest
summaries are tracked under index_batch/results/validation/. E1/E2 timing reports
explicitly identify earlier measurements as historical; post-fix repeats are next.

No build/test/GPU job remains from the negative-control sequence. Next: cached versus
uncached capacities and tails, three-process promising repeats, E1/E2 post-fix repeats,
E3 memcheck/profile; then independent E4 fused N4 and useful combinations. Keep result
and checkpoint commits separate. Goal remains active; no push or production promotion.

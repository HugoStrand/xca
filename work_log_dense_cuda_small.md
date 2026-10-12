# Small-matrix CUDA benchmark and optimization work log

## Current checkpoint — 2026-10-11

Preparation and optimization persistent goals are COMPLETE. The optimization goal
was marked complete after checkpoint0809f85. All priorities E1–E4 have checked prototypes and
measured evidence. E3+E4 is the recommended ordinary-SE experiment; custom pivoted
solve is numerically rejected. Read experiments/dense_cuda_small/COMPARISON.md.

Results18cb4b9; final CPU-only controld013395. Best N4/r16 full128 warm API: order6
6.815–6.869ms, order8 9.135–9.197ms,14.0–14.8× matching CPU in three processes.
Single-index calls still favor CPU. No production promotion, no push. Post-handoff
benchmark evidence includes708 backend/case groups plus4 final CPU-only groups.

All native sequences62308/10148/80198/20942/41151 are terminal. No GPU/build/test job
remains. Recommended E4 six CTest entries pass (four API regressions including dense
complex rotated operators and changed G); API/custom primitive memchecks0 errors.
E3 API/DLR memchecks0 errors. Both corrected E1/E2 suites and repeats pass.
Custom-solve mode fails strict gates as documented; library-control six entries pass,
and its primitive memcheck0 errors. Baseline23 hashes match064b64e. Unrelated user
files remain untouched. Supervisor stays exact-goal-bound; completion must leave it
inactive, never create another exploration goal automatically.

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

## E3 performance checkpoint — 2026-10-11

Code3d7c826, results0d53440 and CSV normalization31e22e7; checkpoint separate.
Twelve E3 runs/290 backend-case groups pass strict parity (sampled errors0): capacities
1/8/32/128 throughput, three fresh128 repeats, uncached metadata, orders through12,
rank26, seed31415 and129-index tails. Six repaired E1/E2 repeats add168 groups.
Full128 batch crossing orders6/8 medians32.727/32.741/32.787 and44.459/44.472/44.500ms;
current CPU95.245/129.747ms, frozen CUDA599.061/831.037ms. First37/50ms and setup7.5ms
still beat CPU at this width. Uncached37/50ms shows metadata preparation cost; tails129
69.684/94.788ms and small padding remain real regressions. No universal speedup.

Nsight audit passes:414 kernels,~31.853ms active,~32.63ms traversal, zero inner
alloc/free/explicit host transfers,6295552D2D and8192UVM bytes. Peak74747573B includes
refs/libs, known arrays23629888B. Geometry attributes26.369ms to44 tiny4×4 products/
vertices (gridZ2048=rB), only1.162ms to40 wide rank transforms. Solves3.57ms (~11%).
Thus E4 should retain pivoted library LU and target tiny products before custom solving.
Raw trace stays in ignored cbuild-small-profile/batch; tracked CSV/geometry evidence.

API memcheck complete0 errors; DLR sanitizer/build/fusion validation remain queued in
80285. Work tree user files untouched, no push. Future checkpoint must clear live-job
entry only after polling its terminal result; maintain source hashes and strict gates.

## E4 numerical investigation checkpoint — 2026-10-11

E3 sanitizer resultf20806e. E4 prototypea30708d with isolated nvcc/CUDA target,
normal FMA, pointwise weighting+parity+accumulation and final convolution add/scale.
Mode0 is library control; mode1 pointwise; mode2 custom N4 products plus pointwise.
One thread/output element,128 threads/block,40 registers for N4,24 weighting and16
add/scale (compiler report retained). Library LU/rank transforms unchanged.

Disabling FMA or reversing complex weighting operands failed high-order parity,
though low-order analytic checks passed. Input-first multiply under normal CUDA FMA
matches the retained DGMM primitive exactly and all6 mode1 CTests pass. Sequential
interleaved complex custom dot also failed high-order gates (rel1.014 at norm2.69e-31);
grouping real dot products using explicit __fma_rn fixes the tested discrepancy and
all6 mode2 entries pass. Negative/positive controls retained, no tolerance changes.
Broader primary benchmark gates are running in62308. Initial B1 order8 mode0/1/2
4.207/4.150/3.359ms, exact reference parity; this remains slower than CPU. Full-batch
custom product benefit and additional rank controls still pending.

## Final exploration checkpoint — 2026-10-11

Implemented/retained separate serial_stream, graphs, index_batch, fused_small and
fused_solve folders. Main code/results18cb4b9, CPU-only controld013395; this plan/log
checkpoint is separate. E4 primary17 runs/250 groups have exact sampled CUDA parity,
including ranks16/26/36/order12/second seed/tails. Dense complex basis and same-address
G changes pass at capacities1/8/128. Three-process B1 andB128 repeats are recorded.
Mode0 library and mode1 pointwise controls show the dominant gain comes from N4
products, with E3 cross-index width providing throughput. Padding/tail/cache-miss
costs are explicit; context (~156ms) is separately recorded, not hidden as API work.
Initialized-process setup/first4.48/10.8–14.4ms; adding measured context estimates
about2 full-batch calls to amortize against CPU, not a full cold-library measurement.

E4 order6 B128 trace396 kernels/~5.9–6.2ms active, no inneralloc/free/host transfers,
44 custom products/vertices~.40ms versus E3's26.37ms,40 rank GEMMs1.16ms, solves~3.6ms.
Known arrays23.63MB, observed peak74.75MB includes references/libs. B1 known221KB,
peak51.31MB,243 kernels. Full API retains two malloc/free calls and benchmark fences;
runtime/geometry CSVs accompany raw samples. No peak-occupancy claim.

Trace-justified solver probe reconstructs at~1e-16 and is faster in isolation, but
fails coefficient/public strict parity. Cached pivots/factors/complex double preserved,
no tolerance adjustment. Public rejection reproduced with source-hashed failure
manifest; library control passes. Micro final CSV measures both reconstruction
residuals; earlier zero library residual placeholder is explicitly identified. Keep
library LU in recommended experiment. Failed controls/prototype are retained for review.

Final archive audit passes:17 accepted fusion manifests,250 groups, zero sampled
parity errors, clean sanitizers, retained numerical rejection. CPU prefix final
backend-cpu smoke passes4 groups. Production hashes unchanged, user work preserved.
No remaining required strategy, combination, report or checkpoint work. Mark the
existing persistent optimization goal complete after this commit and confirm the
watcher observes the completed goal. Do not push or promote automatically.

## Goal closure and watcher confirmation — 2026-10-11

Persistent goal tool confirms COMPLETE (recorded elapsed work6079s, about1h41m).
The watcher next observed no_goal at1791749930.2408204 and exited, as designed when
there is no active goal. Its tmux session is gone; no resumption was queued. Config/
boot registration remain but cannot recreate a completed goal. Final results and
separate checkpoint are committed; production baseline and user files preserved.

## Dense-small implementation goal — 2026-10-12

New persistent goal: deliver plan_dense_cuda_small_generic.md through operational
native dense-small functionality and Solver integration. User requests Sol 6.1 medium;
supervisor continuations explicitly select gpt-6.1-sol / medium / standard speed.
The active setup turn cannot change its own running model; the next supervisor turn
performs the handoff. Use terse house-style [scope] commit messages. Commit plans and
step-by-step logs separately from code. Preserve unrelated work, all experiments and
the library CUDA baseline; do not push.

Plans now require only power-of-two dense Hilbert dimensions from atom_diag. Arbitrary
block sizes and rectangular products remain future block-sparse work, including
parallelism across topologies and shared computation DAG investigation. Python APIs
may change for bounded native work submission and MPI orchestration. GH200-class
hardware is the performance target; A4000 results cannot establish target defaults.

Supervisor: new isolated runtime cbuild-supervisor-small/implementation/, bound to
this thread and exact new goal on dev_cuda. Existing exploration state retained.
All 28 supervisor tests pass, including simulated quota recovery and active-turn
protection. Live read-only check reports thread_active. Detached launcher started;
existing marked @reboot registration now points to implementation/config.json.
No real quota exhaustion/reset was forced; the next natural reset validates live
recovery. Respect user pause, blockers, quota permissions and unknown start outcomes.

Next: verify model handoff and watcher; execute M1 in a new experiment folder with
power-of-two product kernels and strict finite/parity checks, recording measured
specialization/crossover evidence. Then implement M2–M5, including native paired SE,
dynamic interactions, SPGF/correlators, bounded resources, bindings and MPI Solver
integration. Full functionality is the objective, not just a baseline-delegating
facade. Check source manifests before changes; preserve the original baseline while
adding cuda_small and necessary build/Python integration. Qualify only locally measured
policies; document unavailable GH200 validation rather than inventing performance.

## M1 power-of-two screen — 2026-10-12

Progress: isolated power_two experiment committed in81ac16d, source/evidence and
comparison labels in9e6ccd1; additional labelled failure manifests committed separately.
All original experiments and23 protected baseline source files remain unchanged.
New kernels template dimensions2/4/8/16/32/64 with distinct namespace/C ABI and
configurable CUDA architecture. Library LU/rank products retained. Six inherited
CTest entries pass; expanded product/left-right vertex checks pass through64.
A finite-checked device-event primitive benchmark records dimensions2–64 and lanes
1/8/32/128. No production implementation is claimed yet.

Strict full-API screen: N4 custom B128 passes28 groups (orders6/8 crossing6.827/9.170ms).
N8 custom fails but library B128 passes28 groups. N2 B128 fails in both custom and
library controls; serial B1 library passes28 groups. N16 custom fails candidate/CUDA;
serial and batched library controls also expose pre-existing baseline CUDA/CPU
individual discrepancy at order3 (relative2.493e-10). Added comparison labels without
changing numerical gates. Followup custom-only screen labels all N2/N8/N16 failures
candidate/CUDA individual, retained with hashes. Do not weaken tolerance or promote
primitive success as evaluator acceptance. See power_two/RESULTS.md and results/.

Primitive B128 product crossover is around32–64 on A4000 (single-process screen),
not a usable whole-evaluator cutoff given parity failures; GH200 remains unmeasured.
All build/test/benchmark handles from this turn completed; no GPU job left running.
Build: cbuild-experiment-power-two, source ~/env/xca-cuda.sh, mode XCA_FUSED_MODE=2.

Next actions: isolate N2 rank/solve width differences and N8/N16 custom arithmetic
(without weakening gates); test qualified width/kernel combinations and preserve
whole-baseline fallback for rejected configurations. Extend physical fixture coverage
and three-process/rank/seed controls for actual dispatch choices. Then progress M2–M5
at c++/triqs_xca/dense/cuda_small, retaining full native paired/dynamic/SPGF/correlator
scope, bounded resource/submission APIs, Python MPI integration and three build modes.
M1 is incomplete, goal active. Do not mistake this screening report for completion.

## Arithmetic diagnosis and native production foundation — 2026-10-12

Control/evidence commit38d123d: N8 tile4 products match sampled cuBLAS primitives
exactly and current-source full screen passes28 groups, crossing B128 order6/8
20.169/27.724ms (library45.539/61.842). Broader repeats/ranks/seeds pending. Preliminary
tile4 manifest explicitly excluded because its binary predates finalized control
source; source-matched rebuild/run supersedes it. All negative controls retained.
N2 per-lane solves restore parity with wide rank GEMMs; rank-only splitting does not.
RHS groups2/4/8/16 fail strict evaluator checks; group3 fails coefficient checks.
Group1 extended N2/N4/N8 DLR tests pass. Three fresh N2 processes pass8 high-order
case groups each; per-lane solve launch cost leaves it slower than CPU (~95/131ms).

Production code7d00486 adds c++/triqs_xca/dense/cuda_small with owned handles/stream
and full native evaluator API. This is a sequential traversal foundation, not a
completed optimized evaluator. No baseline delegation: native dynamic/paired SE,
SPGF and correlators use the new primitives, explicit upload-event handoff and owned
flat accumulation. Solver status capacity drains safely rather than rejecting long
traversals. Kernel tile4; provisional rank16 N4/N8 custom products, other shapes
library; vertex primitives currently library. Batch work policies remain pending.
Eight reused native CTest entries pass12.24s, including dynamic spin rotation/parity,
correlator statistics, layout and flat-index checks. CPU static library build also
passes and AUTO skips CUDA compiler discovery when NDA lacks CUDA. CUDA-small static
build passes; optionAUTO/ON/OFF and export dependency introduced. No bindings/Python
or installed-consumer claim. Original23 protected hashes still unchanged.

All build/test handles from this checkpoint completed; no GPU work left running.
Supervisor remains active. Next: integrate batched ordinary SE into common native
backend with qualified N2 solve policy/N4-N8 tile4 and conservative library routes;
remove experimental resource limits, implement actual-width tails/bounded caching,
and maintain full native method functionality. Extend build/install/sanitizer tests,
then bindings/Solver bounded MPI APIs and persistence. Expand rank/order/seed/repeat
qualification alongside that work. M1/M2 qualification is still incomplete; goalactive.

## Native batching and bounded ranges — 2026-10-12

Code commitsd341d73/68be63d. Ordinary SE now uses native batches for rank16 N4/N8
with qualified width families8/32/128, then smaller families/serial tails. All chunks
accumulate contributions in caller order into one device output; no host partial
sum changes. Dynamic metadata replaces384/64 prototype limits, three width caches
are bounded, workspace preflight/preparation budgets fall back to native serial
before adding contributions. Full enumeration streams through128 host descriptors;
no order12/4096 limit remains. Strict tests exercise actual batch counters, duplicates,
changed inputs/cache keys, invalid-prefix recovery, empty calls and resident-only
budget fallback. Integration tests caught/fixed physical-pole rescaling and mutable
backbone prevalidation. Qualified custom vertices now serve native serial methods.

All9 native CTest entries pass51.17s including enumeration8192>4096. New half-open
ranges preserve ordinary/paired SE, SPGF and one-time correlator accumulation inside
C++; paired ranges filterfb(0)==0 using global stride, not local range length. Range
checks compare indexed baseline results, empty/uneven ranges, invalid bounds, and
fermionic/bosonic endpoints; batch_api passes39.69s. Public GF aliases support the
upcoming binding generation. Python bindings/MPI orchestration are still unimplemented.

Production screens: N4/N8 each36 backend/case groups pass at orders3/6/8 with
8/128/129 indices. N4 B128 order6/8 ~6.93/9.23ms; N8 ~20.20/27.73ms. Updated custom
serial-vertex N4 screen16 groups passes:128 ~6.88/9.20ms,129 ~9.36/12.63ms, versus
retained padded prototype129 ~17.7/23.9ms. These are single-process local screens,
not final multi-process/GH200 dispatch qualification. Source/binary manifests and
validation logs committed in8245422; log trailing whitespace normalized separately.

Build variants: CPU AUTO static library/new benchmark build and CPU smoke pass;
CPU ON correctly fails requiring CUDA NDA. CUDA OFF library build passes, with all4
actual NDA GPU smoke tests (not disabled cases). CUDA ON restored and batch_api
rebuilt. No install/consumer or missing-compiler AUTO claim yet. All23 original
baseline hashes remain unchanged. No build/GPU handles remain running at checkpoint.

Next priorities: batch paired SE so the Solver's default paired route receives the
throughput gain, extend rank/method qualification, then optional CPU-safe GPU bindings
and Solver backend selection/persistence. Use native ranges for MPI without whole
rank-local index arrays, with collectives/error coordination outside local work.
Complete resource budget/lifetime checks, finite metric-overflow rejection in the
benchmark, compiler-unavailable AUTO/install/consumer tests, sanitizers/profiling and
full CPU/CUDA/Python suites. Target defaults remain provisional without GH200 evidence.
Goal stays active; implementation/qualification is incomplete. Preserve unrelated
files, original experiments, baseline and terse [scope] commits; no push.

## Paired batching and Python Solver integration — 2026-10-12

Codef951c70 batches paired SE using a preserved prefix and both special-vertex line
directions. Shared schedules key ordinary/paired method identity. Half-open paired
ranges skip excluded direction intervals without enumerating them. Strict N4/N8
checks compare retained CUDA results across widths8/32/128, duplicates, tails,
ordinary/paired cache transitions and filtered range boundaries; batch_api passes49.20s.
Dynamic-interaction batching remains pending; its existing native serial path remains.

Code00ead97 adds optional _dense_cuda and _dense_cuda_small extensions, generated
bindings and Solver dense_backend='cpu'|'cuda'|'cuda_small' selection. CPU default
and symmetry selection retained. The library facade forwards arithmetic to the
protected CUDA evaluator and preserves its shared-NDA serialization requirement.
Small exposes atom_diag constructors and native work ranges; C++-only overloads
are ignored only by binding generation. CPU/import availability is governed by a
generated build config, so stale extension files cannot advertise disabled backends.
GPU imports are lazy; archive inspection/copies do not require GPU initialization.
Capacity/workspace preferences survive copies/HDF5; old archives default to CPU.

Small Solver MPI paths use one native half-open range for ordinary/paired SE, SPGF
and one-time correlators, never a full rank-local np.arange. Collectives occur after
local work, with coordinated error flags before output reduction. CPU/library legacy
index-array paths are preserved; their full-local materialization remains a limitation.
No new GPU ordinal selector yet; current tests use the launch environment/default
visible device. Explicit placement/device lifetime coverage remains outstanding.

Six physical Python backend tests pass in serial and on each of two real MPI ranks,
including CPU/GPU method comparisons, proof native paths avoid the array helper,
empty/uneven range partition arithmetic, injected collective failure, self-consistency
and HDF5/copy reconstruction. CPU-only backend/import, h5_io and solver_copy CTests
pass3/3. Original23 protected hashes unchanged; all active build/test handles ended.

Next: dynamic interaction batching and broader rank/method qualification; explicit
GPU placement/lifetime contract, finite comparison-metric overflow rejection, resource
budget hardening, binding regeneration ON/OFF and compiler-unavailable/library-only
validation with the new extensions. Complete installed consumers, sanitizers/profiling,
repeatable real workload timing and full CPU/CUDA/Python suites. GH200 defaults remain
unqualified/provisional. Goal active; do not declare completion from these narrow checks.

## Interaction batching and resource checks — 2026-10-12

Batched backbones now carry n_int so fermionic/bosonic parity and mixed channel
bookkeeping match native serial traversal. A fitted density/spin-channel test exercises
129 ordinary/paired evaluations, proves batches executed, and passes strict retained
CUDA comparison plus CPU diagnostics. Python backend suite passes after this change.
Dynamic fitting itself remains covered separately by existing tests; this new case
intentionally isolates already-fitted channel data.

Benchmark comparison rejects nonfinite norms/errors/relative metrics in addition to
nonfinite values. Negative tests cover NaN, infinity, overflowing finite-component
norms and tiny incorrect zero-reference outputs; no tolerance changed.

Serial vertex metadata now has a budget and drains/evicts completed tables before
reallocation. Correlator resizing checks overlapping replacements and temporary
endpoint/output storage, evicts batch caches if needed, and fails before allocation
when the budget is insufficient. Direct slab downloads eliminate the second full
host output buffer. Table budgets refresh after batch allocation and reserve temporary
storage during correlators. A correlator budget-failure/recovery test passes.

Binding regeneration ON succeeds for both optional extensions and yields no tracked
wrapper differences; normal OFF restored. Native batch_api plus Python backend tests
pass2/2 in51.12s after regeneration; all jobs ended. These checks do not replace full
CPU/CUDA suites, sanitizers, installed consumers or target-hardware qualification.
Original23 source hashes remain unchanged; unrelated work preserved.

Next: broaden dimension/rank/method/seed qualification and performance repeats,
explicit GPU placement/lifetimes, allocation/metadata lifetime audits, compiler-missing
AUTO and library-only extension checks, install/consumer verification, sanitizers and
profiling. Run full CPU/CUDA/Python and MPI suites after the remaining changes. Target
GH200 defaults remain provisional. Goal active; no completion claim or push.

## Full suites, installation and memory validation — 2026-10-12

Full CUDA build and75/75 CTest entries pass120.18s. Full CPU build and67/67 entries
pass48.78s; CPU GPU-disabled tests remain disabled and do not imply GPU coverage.
An isolated CMake fixture verifies AUTO disables custom kernels when the CUDA compiler
is explicitly unavailable; actual library-only runtime support was verified earlier.

Staged install at /tmp/xca-small-stage builds/runs a fresh external C++ consumer,
including exported target dependencies, native constructor and range evaluation.
Installed Python initially failed because installation stripped runtime paths and
libcutensor.so.2 was not in the loader cache. Fixed CMAKE_INSTALL_RPATH_USE_LINK_PATH
so external dependency directories survive install; staged Python import and7 backend
tests now pass. This fixes a packaging fault rather than requiring an extra environment
workaround. Normal CUDA install prefix restored to ~/apps/xca-cuda after staging.

Compute Sanitizer memcheck/leak-check on paired batching/cache transitions, one-time
ranges and correlator budget recovery:3 GoogleTests pass;0 errors,0 bytes/allocations
leaked. Scope is explicit in the retained log, not a blanket sanitizer claim for every
path. Validation logs/consumer/no-compiler fixtures and source/binary hashes retained
under benchmarks/dense_cuda_small/results/system_validation; whitespace normalized.
Original23 protected source hashes remain unchanged. All live handles from this turn
finished; no GPU/build job left running.

Next: explicit GPU placement/device lifetime contract; broaden rank/dimension/method/
seed qualification and repeatable real-workload timings; profiling/residency audit,
resource accounting/lifetime review and library-only optional-extension tests. Target
GH200 hardware remains unavailable and target defaults provisional. Goal active;
full-suite success does not prove every plan requirement or end the implementation.

## Process device placement — 2026-10-12

Supervisor configuration verified: gpt-6.1-sol, medium effort, standard service tier;
its log confirms this thread is active. Persistent implementation goal remains active.
The supervisor selects these settings on resumption; this check alone does not verify
or change the model serving the current turn.

Solver now accepts gpu_device=None or a nonnegative visible ordinal, selects it before
GPU evaluator construction, and preserves the preference in copies/HDF5. None respects
the current CUDA device and CUDA_VISIBLE_DEVICES. Subsequent initialization rejects a
conflicting process device or an externally changed current device. CPU imports remain
independent of the optional CUDA runtime bindings. Old archives default to None.
Bindings expose device count/current/select functions in both optional modules.

Initial generation omitted functions because c2py matches fully qualified names with
a leading ::; corrected filter and regenerated both wrappers. Nine backend tests pass
on CUDA; all nine pass on each of two real MPI ranks sharing the single A4000. CPU
backend test entry passes after refreshing configured build-tree Python sources (GPU
cases skipped). Mocked device-conflict checks supplement actual ordinal selection;
they do not establish multi-GPU behavior. Baseline verifier reports all23 files unchanged.

This defines one unchanged device per process as a caller requirement. Native C++
entry-point/destruction guards and checks for an external device switch between
initialization and evaluation are still pending, as are multi-device hardware tests.
Continue with that lifetime audit, then rank/dimension/method/seed qualification,
repeatable workload timings/profiling and library-only optional-extension validation.
GH200-class defaults remain provisional. Code and this checkpoint are separate commits.

## Native device lifetimes — 2026-10-12

Native owners now record their construction device. Evaluator uploads and correlator
work reject a changed current device before allocation/submission; stream completion
and default-upload handoff also enforce that device. Resource owners temporarily
select their device for destruction and restore the caller selection only after all
member buffers/handles are released. Context members are declared first to outlive
those resources. One instance still requires serialized calls.

New tests interleave independent evaluators with different batch widths and changed
inputs, then destroy one on another host thread and continue using the other. Focused
native batch_api/Python entries pass2/2 in52.50s. CPU compatibility entries pass2/2;
the CPU native test remains disabled. Compute Sanitizer memcheck/leak-check on the
new lifetime tests passes one actual test with0 errors and0 bytes/allocations leaked.
The two-visible-device regression is skipped on the single A4000; it covers rejected
work after a device switch and caller-device restoration following destruction, but
is not locally validated. No multi-device hardware claim is made.

Code committed as503c260 separately from this checkpoint. Native ordinary/paired
rank16 N4 throughput qualification with seed31415 has started under the existing
strict harness. Wait for its tracked process before any other GPU benchmark. Then
repeat in fresh processes, screen higher ranks/dimensions, and retain capability and
crossover evidence. Library-only optional extensions and broader profiling remain
pending. GH200 hardware/default qualification remains unavailable locally. Goal active.

## Repeated workloads, residency and library-only Python — 2026-10-12

Six fresh native benchmark processes (three each N4/N8, seeds31415/1729/31415)
pass strict individual/aggregate CUDA gates for ordinary/paired SE, crossing,
rank16, orders6/8, counts128/129, samples3. N4 B128 medians6.875–9.284ms;
N8 medians20.172–28.253ms across methods/orders. CPU speedups roughly13–15x
for these fixtures. Raw data/provenance retained; report in
benchmarks/dense_cuda_small/NATIVE_RESULTS.md. These records precede storage reuse.
Recorder now includes the native entry source; earlier entry hashes transparently
backfilled from each recorded revision. High-order CPU error remains diagnostic.

Timed-region Nsight profiling found two transient4096-byte device allocations in
warm ordinary SE: upload assignment replaced storage, and NDA regular-array host
conversion copied the device result by value. Changed both to view-based operations.
Strict batch_api/Python tests pass; updated warm capture has no CUDA allocation/free
APIs, one propagator upload and one result download. Solver status still causes one
4096-byte unified migration in each direction. Scope: N4/order6/128 ordinary SE only.
Saved cold/warm runtime/kernel/copy/allocation tables and provenance; do not infer
allocation freedom for every API/shape. Code6a2f143 separate from evidence/docs.

Separate cbuild-library-cuda uses XCA_CUDA_SMALL=OFF, no CUDA compiler configured,
Release clang19 and source links to existing dependency checkouts. Full build passes;
library-only Python and NDA GPU smoke entries pass2/2. Staged package import at
/tmp/xca-library-cuda-stage exposes retained CUDA and reports Small unavailable.
It must be refreshed for the later default-device serialization repair below.

Full CPU/CUDA suites exposed gpu_device=None failing HDF5 writes in existing archive
and dynamic tests. Focused earlier test used explicit ordinal0 and missed the default.
Omit the default field so constructor restoration supplies None; add default archive
round-trip regression. Repair b0020e2. Focused archive/backend/dynamic entries pass
in both trees, then full suites pass67/67 CPU(48.05s) and75/75 CUDA(112.36s).
Failure and repaired logs retained. No tolerance changed; protected baseline23 files
remain the comparison control.

Native rank experiment in experiments/dense_cuda_small/native_ranks generates a
separate namespace from the current native code and changes only N4/N8 rank eligibility.
ON/OFF product controls and an actual batch-counter assertion prevent mistaking serial
fallback for a batching result. Snapshot generator/entry/code are hashed by the runner.
First rank26 N4 screen is running (tracked exec session86664); wait for terminal state
before any other GPU work. Production remains rank16-only until qualification/repeats.

Next: finish rank26/36 screens and broader dimension/method coverage, refresh library-
only/install Python after archive repair, split memory reporting into owned device,
host snapshots/metadata and opaque CUDA-library allocations, then completion audit.
Supervisor process219671 verified live with the implementation config; model/effort
still Sol6.1 medium for resumption. Multi-GPU and GH200 tests remain unavailable;
local A4000 evidence is provisional. Goal active; unrelated user files preserved.

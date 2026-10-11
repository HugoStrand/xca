# Plan: small-matrix CUDA optimization experiments

Status: exploration complete,2026-10-11. E1–E4 have checked prototypes and measured
results; E3+E4 combinations are measured, and the trace-justified custom-solve follow-up
is numerically rejected. [Comparison/recommendations](experiments/dense_cuda_small/COMPARISON.md)
and [final checkpoint](work_log_dense_cuda_small.md) record the outcome. No experiment
was promoted; production064b64e and all23 protected hashes remain unchanged.

## Objective and limits

Improve dense C++ evaluation of N=4 at higher expansion orders, especially orders6–12,
using a few topology/index selections. Optimize complete-call latency for one index
and throughput for batches separately. More available GPU FLOPs do not guarantee a
speedup for tiny matrices: launching many short operations and repeatedly waiting
can dominate; the RTX A4000's precision characteristics also constrain FP64 work.
Success is measured against the existing CUDA evaluator and the CPU on identical
workloads. No promise to beat CPU for every case or to saturate peak GPU FLOPs.

Keep complex double, LU-based DLR conversion, operand/sign conventions and current
host-returning APIs. Experiments may initially support ordinary self-energy only,
with explicit rejection of other methods. Promotion requires the reused analytic,
rotation, dynamic-interaction and correlator checks for every claimed path. Python,
solver changes, block-sparse CUDA and production dispatch selection remain outside
scope. No experiment is automatically merged into the production evaluator.

## Benchmark contract

`benchmarks/dense_cuda_small/README.md` documents the runnable harness and profiles.
Use fixture dense-small-v2, N4/rank16 with rank26 control, orders1/3/6/8 and bounded
order10/12 coverage. Two directly constructed topology families, crossing and chain,
exercise different special-vertex positions/traversal lengths. No exhaustive topology
or flat-index enumeration. Seed1729 and explicit index lists make cases reproducible;
repeat promising results with a second seed and ranks16/26/36.

Quick mode samples1/8 indices; throughput mode1/8/32/128. Sweep batch capacity B
independently of selected-index count in an experiment, including tails B+1 and
nonmultiples. Bounded test count does not limit the evaluator's long-index support.
Use se/pairs/spgf coverage where implemented. Pair counts are seeds, not ordinary
index counts. Include duplicate indices and maintain caller multiplicity/order.

Record raw samples, medians, min/max/MAD, first call, constructor cost, owned memory,
exact command, revision, source/binary hashes and loaded libraries. Fresh timed
objects prevent correctness checks from concealing graph/schedule preparation.
Complete API time includes transfers, host scheduling, synchronization and owning
host result materialization. Device-only timings are diagnostic and cannot replace
this metric. No concurrent benchmark/build/GPU workloads. At least three fresh
processes for any promoted claim, same thread settings/prefix/GPU, alternating backend
order. Save failed/negative experiments as well as wins.

Each candidate is compared with the unchanged CUDA implementation per index and per
summed result at relative1e-10 with a negligible1e-280 absolute floor; zero substitution
must fail even for tiny results. Changed input at the same address, empty and duplicate
index vectors are checked before timing. CPU comparisons remain required at orders1–3.
At higher orders report CPU errors and reference norms separately: initial probes
found ~4.1e-7 relative error for an order8 contribution of1.18e-18 and ~0.25% for an
order12 SPGF contribution of2.65e-41. This is a baseline numerical limitation, not an
optimization tolerance allowance. Do not silently relax the parity gate to make a
strategy pass. Investigate failed lanes, operation order and cancellation; retain
existing unit-test tolerances. Accuracy work is a separate investigation if needed.

## Measured starting point and review revisions

Seven validated baseline runs and the N4 profile are recorded in
[benchmarks/dense_cuda_small/RESULTS.md](benchmarks/dense_cuda_small/RESULTS.md).
All224 backend/case groups pass CUDA-reference parity. N4/r16 crossing single-index
median API time is4.43ms at order6 and6.27ms at order8, versus CPU.75/1.03ms.
At128 indices order6 costs599ms and order8 costs831ms: current vector submission
provides essentially no per-index throughput gain. Three process repetitions have
small spread relative to the proposed20% improvement threshold.

The order6 trace records257 kernels and about335 device synchronizations per call.
Traversal active kernels total2.814ms under profiling (GEMM1.348, solves1.251),
against5.53ms profiled traversal wall time. Twenty LU conversions and ten convolutions
make DLR work central even though matrices are4x4. Inner traversal is allocation/copy
clean; complete API ranges have two malloc/free calls each, whose ownership should
be identified before choosing reusable staging. These findings prioritize E1 and E3;
E2/E4 test complementary launch/traffic reductions. Profile overhead prevents treating
the kernel-time sum as an unprofiled performance bound.

Review changes incorporated: fresh timed objects prevent validation from hiding graph
setup; individual contributions and changed input guard against invalid speedups;
measured CPU-relative errors at high order are distinct from baseline parity; protected
source hashes enforce isolation; shared-LU wide RHS batching is favored over duplicated
factorizations; pointer-table lifetimes and solver status are explicit prerequisites
for asynchronous execution. Graph replay must include amortization and cache misses.

High-order accuracy needs particular care: coverage found a lane with1.35 relative
CPU discrepancy in an order10 case, with all lane absolute differences in that case
<=3.54e-34. Very small cancelling lanes can be dominated by floating-point effects.
Do not optimize numerical artifacts or silently weaken the gate. If reordering changes
such lanes, classify the prototype as failing current parity until a targeted accuracy
investigation supplies stronger evidence. Performance success and physical accuracy
are separate conclusions. Rank26's sampled maximum per-index CPU relative error was
3.37e-9, motivating rank and conditioning controls.

## Strategy sequence

### E0: profile and establish the control

Use the committed baseline and a small order6 N4/r16 trace, restricted to measured
calls with `--within bench.cuda.sample`. Record kernel counts, active kernel time,
CPU wall time, synchronization/API calls, allocation/copy events and managed metadata
migration. Measure B1 and throughput scaling before modifying code. Use the existing
larger-matrix data only as context, not as a substitute for N4 evidence.

Add `cpu_hoist_control/` as a bounded CPU experiment if CPU comparison is material:
compute convolution H*F and H*G once outside the time loop, leaving production CPU
unchanged. The old benchmark showed an algorithmic gain from this alone at larger N;
measure it at N4 before attributing all gains to device execution.

### E1: reduce synchronization and host metadata overhead

Folder `serial_stream/`. Copy only needed evaluator/backend code into an experimental
namespace with a provenance note. Own a nonblocking CUDA stream, cuBLAS/cuSOLVER
handles, persistent solver status and workspaces. Enqueue the same operations and
synchronize at the public return/error boundary; do not change arithmetic initially.
Benchmark B1 to isolate host launch/wait cost from cross-index batching.

Removing `DeviceOps::finish()` alone is unsafe and incomplete: NDA wrappers may
synchronize independently, and the NDA solver reads shared managed status immediately.
Use owned adapters or a demonstrably safe scoped setting that restores shared state;
never globally disable solver synchronization. Keep statuses alive and inspect them
only after completion. Also address pointer-table reuse: current host code overwrites
unified tables before the next GEMM, safe only because earlier calls finished. Use
immutable per-schedule device tables or stream-ordered updates with valid lifetime.
Do not assume unsupported zero-stride batch broadcasting.

Precompute topology/index scalar schedules and operator/kernel selectors once per
call or cache them by content with bounded ownership. Transfer small metadata once,
keep work tensors resident. Measure schedule-build cost and cache hit/miss cases;
propagator content must never be assumed unchanged by its address.

### E2: CUDA Graph replay

Folder `graphs/`, explicitly based on the validated E1 adapter when appropriate.
Capture or construct the device schedule on the owned stream; upload current G and
metadata outside the graph or through stable owned transfer nodes. Reuse the graph
only for compatible topology, indices, shapes and operator tables. Measure capture,
instantiation/upload, first launch and warm replay, and compute amortization count.
Graph rebuilds on changing indices/topologies are part of the cost model.

Graphs reduce repeated host submission overhead but do not increase arithmetic
intensity or fix underfilled kernels. They may therefore help B1 without enabling
full GPU throughput. Stream capture forbids captured-stream/device synchronization,
and cannot use the legacy NULL stream; wrapping the current public evaluator directly
is not valid. See [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
Pointer and workspace addresses must remain valid. Host-mode scalar arguments can be
captured by value: changing host variables later is not automatically a graph update.
Bound cache memory and verify changed-data/index behavior under the harness.

### E3: batch independent indexed evaluations

Folder `index_batch/`. This is the leading throughput hypothesis, and must be tested
even if E1/E2 help. Process B lanes with the same topology and traversal stage. Lane
metadata carries each index's directions/poles/orbitals/parities. All directions
follow the topology's same operation schedule; encode signs/selectors as data instead
of launching a separate host traversal per lane. If any operation schedule truly
varies, group compatible lanes explicitly and include grouping cost.

Start with B=1/8/32/128, later256/512 if evidence warrants. A useful candidate layout
is `(r,B,N,N)`: r*B contiguous N×N products become one larger matrix batch, while
rank transforms act on `(r,B*N*N)` and all lanes share one r×r DLR LU factorization.
A single solve with B*N*N RHS columns can amortize overhead without duplicating LU.
Compare layout/packing alternatives empirically; batched LU solvers are also possible
but duplicate factor/pivot metadata unnecessarily when the coefficient matrix is
shared. [cuBLAS getrsBatched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getrsbatched)
is specifically intended for small systems; its layout, pivot and host-info contract
must be checked if used.

For N4, r16–36, DLR rank transforms/solves scale O(r²*N²), versus O(r*N³) for tiny
vertices: the wide RHS/rank operations may benefit more than N×N products. Batching
primarily increases work per launch and occupancy; independent matrices alone do not
increase each matrix's intrinsic arithmetic intensity. Shared model tables and reused
G can improve effective memory reuse. Preserve left/right noncommuting order.

Retain special-vertex kappa reuse per lane; broadcast G safely, use lane-specific
operator pointers or gather into persistent buffers, and fuse weights where feasible.
A buffer costs16*r*B*N*N bytes: N4/r26/B128 gives851,968 bytes, so a dozen such buffers
is manageable on this GPU. Inventory total live scratch/Tkaps/tables/solver workspace,
check all dimensions and retain a bounded chunk capacity. Batching must not allocate
inside traversal. Report observed peak, not just this estimate.

Accumulate lane results deterministically, preserving duplicates and signs. Compare
sequential accumulation in caller order with a tree reduction only as a separate
numerical change. Tail lanes must be masked/zeroed and never leak previous results.
B1, B+1, changed batch sizes, >INT indices, invalid-prefix recovery and empty vectors
are required. Extend pairs/SPGF only after ordinary SE is correct and measured.

### E4: fused kernels for N=4 and small DLR ranks

Folder `fused_small/`, with a separate CUDA-language target local to the experiment.
First fuse matrix multiply plus per-time weighting/scaling, accumulation and/or
special-vertex contraction to reduce both launches and global scratch traffic.
Use register/shared-memory tiles for complex-double4×4 blocks. Batch across index
lanes and DLR nodes; choose coalesced layout and measure occupancy/register pressure.
Keep library LU solves initially. This isolates whether tiny GEMM/pointwise overhead
is the bottleneck before implementing a solver.

Next, if traces justify it, prototype a pivoted triangular solve specialized for r
and many RHS lanes using the cached LU/pivots. Keep LU semantics; do not replace
conversion with an explicit inverse just to turn it into GEMM. Test reconstruction
and cancellation-sensitive inputs. A fused edge or whole-backbone persistent kernel
is a later bounded probe: it risks register/shared-memory exhaustion, poor occupancy
and serial dependency bottlenecks. Stop at the simplest measured improvement.

Custom kernels may reduce launches and traffic while increasing arithmetic intensity
through intermediate reuse. They cannot remove the mathematical serial dependency
between edges of one index. Cross-index parallelism supplies independent work.
Do not use reduced precision, TF32 or Tensor Core claims for complex-double work
without changing the stated accuracy contract; that change is outside this exploration.

### Lower-priority alternatives

A small multi-stream index pool may overlap independent kernels when one lane
underfills the GPU, but adds handles, workspaces and scheduling; compare after E3
if needed, with independent solver status. Algebraic prefix reuse across related
indices could save repeated edges but changes scheduling/cache complexity; only
investigate if traces show significant repeated work. Operator sparsity/permutation
specialization risks optimizing the occupation-basis fixture alone: verify rotated
complex operators before claiming a general dense speedup. Keep every alternative
in its own folder and report why a deferred strategy was not justified.

## Experiment layout and evaluation rules

Each folder owns its implementation namespace, target/build instructions, supported
methods, README, original baseline revision, measured result files and conclusion.
Reuse `harness.hpp` for all inputs/timing/validation. Optional CMake/CUDA setup lives
under the experiment tree and does not change the production CUDA option or source.
Use isolated build directories, retain the production CPU/CUDA builds, and verify the
23 protected source hashes before every benchmark. Combine winning strategies in a
new folder that names its parent experiments rather than overwriting their results.

For each priority strategy E1–E4: implement a bounded prototype, pass primitive and
applicable C++ comparisons, run quick + relevant throughput cases, repeat promising
wins in three processes, and capture a focused allocation/copy/kernel trace. Save
negative results. A strategy may be rejected on a concrete compile/API feasibility
probe or measurements; a speculative paragraph alone does not count as evaluation.
Any prototype that relies on stale inputs, hides setup, uses wrong signs/layouts,
skips precision checks, or round-trips large tensors fails regardless of speed.

Initial success criteria: repeatable >=1.2x median API improvement over CUDA baseline
on at least two N4 higher-order cases, with the change larger than observed timing
spread; report CPU ratio as well. A stronger throughput target is N4/B>=32 beating
CPU and improving per-index time materially over B1. These are targets, not promised
outcomes or reasons to suppress negative results. Report regressions and memory costs.
If first-call latency regresses, state the number of reuses required to recover setup.
Stop extending a strategy after a bounded correct prototype demonstrates no credible
benefit, or a measured feasibility barrier prevents the hypothesized advantage.

Final deliverable: comparison table of supported cases, single-index latency and
batched throughput, setup amortization, correctness scope/CPU numerical discrepancy,
residency/memory evidence, and recommendation for which experiments merit review.
Leave production baseline unchanged. Complete exploration when every priority strategy
has a measured result or concrete rejection, the useful combinations have been
checked, and the report/checkpoint are committed; peak GPU utilization is not a
completion condition. Keep plan/log commits separate, preserve user work, do not push.

## Review checklist before handoff

Confirm final baseline hashes and samples, high-order numerical diagnostics, honest
first-call costs, exact index lists, and serial measurement execution. Verify the
plan addresses LU/RHS layout, metadata lifetime, graph invalidation, batching tails,
reduction order, memory bounds and the fixed complex-double scope. Then complete
preparation, set subsequent thread turns to Sol6.1/medium/standard, create the separate
exploration goal, rebind quota recovery using a new runtime directory, and record the
precise first experiment. Do not start optimization code in the preparation phase.

## Handoff acceptance

Self-review completed against the code and measured trace: phases/case keys and costs
are explicit; all required experiment families have hypotheses, implementation bounds,
validation and stop criteria. No independent sub-agent review was used. Initial action
for Sol6.1/medium: implement the bounded ordinary-SE `serial_stream/` prototype and
run the quick cases; then pursue `index_batch/` even if graph/stream work is favorable.
Use E1 as the graph prerequisite, record each strategy independently, and follow the
persistent exploration goal until its measured comparison report is complete.

## Recorded outcome — 2026-10-11

E1 owned stream and E2 content-keyed graph replay reduce submission overhead but
remain slower than CPU for single small matrices. E3 shared-LU cross-index batches
reach about2.9× CPU throughput at128 indices. E4 grouped real FMA products plus E3
reach14.0–14.8× matching CPU at N4/r16/orders6–8 in three processes. Pointwise fusion
alone is under2%; no material independent win claimed. Single calls still favor CPU.

Strict parity, high orders, ranks16/26/36, tails, duplicates/empty/changed inputs,
long indices, dense complex basis tests and applicable C++/sanitizer checks pass for
the recommended ordinary-SE prototype. A pageable-upload race was corrected using
an event handoff; rejected arithmetic controls remain recorded. Full API/setup/first/
context costs, cache/padding cliffs, memory and kernel/copy traces are documented.

After custom products, solves dominate. The planned conditional triangular-solve
probe uses cached factors/pivots and reconstructs accurately, but public parity fails;
library solves remain. Optional lower-priority work is deferred. Next work needs
review: adaptive tails/schedule cache, ordered accumulation, reusable staging, and
numerically verified solve improvements. Production/promotion is outside this goal.

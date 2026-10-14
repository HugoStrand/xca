# Bounded scalar self-energy path scheduler

Performance target: GH200-class hardware. This private E4/E5 candidate batches
complete scalar self-energy paths, retaining general evaluator execution for other
block shapes and all correlators. Production classes/bindings are unchanged.

Eligibility requires every propagator block to be 1x1 and each symmetry set to have
one field. This is an initial operation-cohort capability, not a reduced evaluator
API: unsupported cases execute the unchanged general path. Arbitrary scalar sets
with multiple mu/kappa members need separate scheduling qualification.

The generated class reuses current reference code, inserting a scalar worker and
changing the self-energy visitor. Each cohort has at most 512 copied Backbones and
path descriptors; owned NDA arrays/vertices survive reuse of the borrowed visitor.
Cohorts split when special-vertex positions differ and retain discovery order.
Packed cached-LU solves/rank transforms come from the qualified scalar-edge component.
No custom solver/inverse or path reassociation is used.

Vertex application, pole weights, convolution, special kappa/weight/mu operations,
prefactor and sign execute in the original sequence. Each completed lane receives
its own max-absolute cutoff (>1e-16); accepted lanes accumulate in original order
into their output blocks. Unused tail lanes stay zero and are never accumulated.
Input G is gathered through current device buffers, not cached across changed calls.
Weights, descriptors, transfers, host cutoff scan and ordered fan-in are all charged
to complete-call timing. Batch storage is created lazily, avoiding construction for
zero-survival windows. The additional conservative workspace is admitted/reported
before input upload; host metadata/opaque allocations are not complete peak admission.

The private class exposes the same API. Its test executable runs the unchanged eight
CUDA evaluator tests against it. Four independent qualification tests also cover multi-cohort/tail range sums,
changed data/masks, adjacent NCA cutoff values, additional-workspace rejection
before submission, empty ranges and multi-field scalar-set general execution.
Additional multi-cohort/tail benchmarks compare CPU and require exact GPU replay. No Python/MPI selector exposes this candidate.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_scalar_path_tests \
 block_sparse_scalar_path_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild-cuda \
 -R '^experiments/block_sparse_cuda/scalar_paths$' --output-on-failure
python3 benchmarks/block_sparse_common/run_calls.py \
 --cpu-binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/dlr_hoisted/block_sparse_cpu_hoisted_benchmark \
 --cuda-binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/scalar_paths/block_sparse_scalar_path_benchmark \
 --machine thinkstationP620_RTXA4000 --campaign scalar-path-01
```

Generic definitions stay here; shared hardware/campaign results use the benchmark
backend folders, with paired comparisons in block_sparse_common. The retained native,
library, unchanged/matched CPU and edge experiments remain available. Full-candidate
CPU batching, irregular special sets/correlators, higher-order cutoff-adjacent paths, wider shapes,
changed-address ownership and target dispatch remain pending. Selected-index speedups
do not establish complete Solver/topology-sum throughput or N=14 qualification.

## Matched full-call CPU control

`block_sparse_cpu_scalar_path_benchmark` also builds in a CPU-only tree. Its host
scheduler shares recipe.inc with the GPU candidate: scalar eligibility, copied
metadata/cohorts, vertex/edge/special operations, prefactors, cutoff and ordered
fan-in. CPU uses cached LU with packed RHS and hoisted rank transforms. No GPU
arithmetic or native compiler is required for this control. Unsupported shapes and
other observables use the unchanged CPU evaluator (not hoisted/batched controls).

Use this executable as `--cpu-binary` in the paired recorder, campaign
`scalar-matched-01`. CPU rows have no path-count observer/device counters. Shared
recipe extraction leaves the GPU qualification suite unchanged; per-case CPU results
are screened against original CPU and repeat exactly. Compare identical eligible
scalar self-energy rows before attributing any old-baseline speedup to GPU hardware.
Descriptor setup, allocations, host transfers and cutoff/fan-in remain relevant
inclusive costs. Results under the common hardware folder document GPU-losing cases.

## Output-owned accumulation variant

`block_sparse_scalar_fanin_benchmark` and `block_sparse_scalar_fanin_tests` link an
isolated variant of the same evaluator. The host per-lane cutoff scan is unchanged;
surviving lane IDs are grouped by output, preserving original lane order. One GPU
thread owns each output/rank element and sequentially adds all its terms to the
existing value. No atomics, tree sum or algebraic reassociation occurs. Temporary
pointer/offset/lane tables remain owned until stream completion. Their preparation,
upload and kernel completion enter full-call timing; the conservative batch bound
has room for these bounded metadata arrays.

The serial variant remains a control. Tests reuse all 12 semantic/physics cases,
plus a cancellation-sensitive order test with interleaved owners and existing sums.
Use the fan-in binary as the CUDA recorder argument, campaign `scalar-fanin-01`,
with the matched CPU scheduler. A lower latency versus serial GPU does not imply a
win over matched CPU or select a production/GH200 default. No bindings expose this
experimental variant.

## Contiguous scalar storage and scoped profiling

`block_sparse_scalar_arena_benchmark/tests` retains ordered accumulation and places
all scalar input/output blocks in two owned contiguous tensors, with nonowning block
views used only while their owners remain alive. Live-mask handling is unchanged;
input packing, single upload/download, CPU result construction and every scheduler
cost enter full-call timing. General shapes/correlators retain the original route.
The underlying payload admission is unchanged; this removes allocator/transfer call
multiplicity rather than claiming complete opaque/host peak qualification.

`profile_fanin` and `profile_arena` capture only the common runner's second self-energy
call with cudaProfilerStart/Stop. Use Nsight `--capture-range=cudaProfilerApi`,
`--capture-range-end=stop`, `--trace=cuda`, `--sample=none`, `--cpuctxsw=none`.
Profiler wrapper durations include capture/export effects and are excluded from
benchmark comparisons. Source frontends, report hashes and API-count summaries
belong in hardware folders. Timing uses the ordinary arena binary in a separate
isolated paired campaign, with matched CPU and prior controls retained.

## Path-constant coefficient variant

`block_sparse_scalar_constants_benchmark/tests` retains contiguous scalar storage
and ordered fan-in. Vertex coefficients, special endpoint coefficients, prefactors
and signs are computed once per lane. A broadcast multiply uploads one coefficient
per lane and repeats it across DLR nodes, retaining the same complex multiply and
operation sequence. Node-dependent pole/hybridization weights retain their full
node-by-lane representation. Nothing is cached across calls: changed path metadata,
operator values and propagators retain their existing lifetime and update rules.

`block_sparse_cpu_scalar_constants_benchmark` applies the same once-per-lane host
preparation in the matched CPU scheduler and builds without CUDA. The older CPU,
arena and per-block controls remain separately selectable. The new GPU workspace
fits the existing conservative bound, with smaller coefficient uploads. Qualification
includes exact broadcast-versus-expanded GPU products, in-place updates and kernel
tails, plus the existing full evaluator, cutoff, ownership and fallback controls.
Only unprofiled isolated paired runs support performance comparisons; GH200 remains
the target and local timings do not select a production dispatch policy.

## Active high-order arithmetic control

`ActiveOrderSixCutoffAndReallocatedInputs` finds a structurally active index with a
bounded CPU metadata search, then asserts nonzero arithmetic at ordinary amplitude.
It brackets the completed-path cutoff with 24 bisection probes and checks both sides
with a stated relative amplitude margin of 1e-4. It also retains the old input owner
while evaluating separately allocated, changed inputs and then reuses the old owner.
This is distinct from the target benchmark's zero-survival order-six window.

The independent long-double scalar oracle in `precision.hpp` uses qualified path
descriptors and the same cached double LU/rank/kernel constants. It diagnoses
arithmetic; it does not independently validate connectivity or DLR approximation.
At the sampled order-six index, both general and batched GPU arithmetic exceed the
older 1e-13 comparison bound (about 4.28e-13/3.90e-13 relative to the oracle locally).
The new case checks each route independently at 1e-12, keeping all older tolerances
unchanged and requiring contributing-path count agreement. ULP-adjacent high-order
cutoff guarantees, more topologies/poles and target validation of this new test remain
open. No production promotion follows from this bounded qualification.

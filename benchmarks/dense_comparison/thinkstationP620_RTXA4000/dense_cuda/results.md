# Dense CUDA port: results and acceptance audit

Verified on ts, 2026-10-10. The C++ CUDA evaluator implements ordinary and paired
self-energy, single-particle Green's functions, one-time correlators and dynamic
interactions. It preserves the host compute/count interfaces and uses complex
double, persistent device tensors, cuBLAS and LU-based cuSOLVER DLR conversion.
Select `triqs_xca::dense::cuda::DiagramEvaluator` explicitly from
`triqs_xca/dense/cuda/diagram_evaluator.hpp` in a CUDA-enabled build. Calls, including
other NDA solver use in the same process, must be serialized. Model inputs are frozen
at construction; propagators and external operators are read anew on every call.

## Synchronized API measurements

[comparison.csv](comparison.csv) contains 300 rows from ten runs: N=2,4,16,32,64,128
at cutoff 40 (rank 26), and N=32,64 at cutoffs 10/120 (ranks 16/36).
AMD Threadripper PRO 3945WX, RTX A4000 16 GB, CUDA 13.4, driver 580.178.04,
clang-19 Release, TRIQS 4.0. Both thread settings are one. Identical complex
propagators, two poles, occupation operators and evenly spaced index subsets are
used for the CPU and CUDA backends in the CUDA-enabled executable. Beta=2,
eps=1e-12, n=log2(N). Construction/context and first calls are separate rows.
The table averages three subsequent calls, including transfers and host scheduling,
for order three with eight selected indices; speedup is CPU time / CUDA time.

| N | Self-energy CPU ms | CUDA ms | Self-energy speedup | Single-particle speedup |
|---|---:|---:|---:|---:|
| 2 | 3.45 | 17.86 | 0.19× | 0.19× |
| 4 | 6.60 | 19.84 | 0.33× | 0.30× |
| 16 | 70.21 | 26.71 | 2.63× | 2.59× |
| 32 | 284.59 | 43.37 | 6.56× | 6.98× |
| 64 | 1340.13 | 156.27 | 8.58× | 9.80× |
| 128 | 12125.99 | 691.29 | 17.54× | 24.27× |

Paired order-three speedups range from 0.22× (N2) to 15.15× (N128). A paired seed
evaluates both directions, so its count is not the same amount of work as an ordinary
index. At N32, order-three self-energy speedups at ranks 16/26/36 are 4.08/6.56/9.07×;
at N64 they are 5.90/8.58/23.51×. See the CSV for order-one/two results and first-call
costs; the crossover depends on the operation and rank. Small cases are limited by
launch/synchronization overhead. No automatic backend choice was added.

All ten runs passed finite CPU comparisons; worst maximum absolute difference was
5.52e-15. This is a bounded synthetic evaluator workload with three warm samples,
without confidence intervals or solver-wide performance claims. Measurements preceded
the final diagnostic phase ranges; the numerical algorithm is unchanged.

### CPU baseline and algorithm effects

The current CPU convolution repeats rank transforms inside its time loop. At N32,
current / hoisted CPU / CUDA convolution times are 6.98 / 1.73 / 0.47 ms; at N64,
30.09 / 10.30 / 1.58 ms; at N128, 161.53 / 55.38 / 7.90 ms. Hoisting alone gives
about 2.9–4.0×; CUDA then improves on that hoisted implementation by about 3.6–7.0×.
Whole-evaluator speedups include both changes and cannot be attributed solely to
GPU offload. The production CPU evaluator was preserved.

An important control is [cpu_prefix_check.csv](../../../dense_cpu/thinkstationP620_RTXA4000/cpu_prefix_check.csv): the CPU-only
prefix at N32 measured 566.21 ms for order-three self-energy, versus 284.59 ms for
the host evaluator in the CUDA-enabled run. Its convolution timings agree closely
(7.01 / 1.75 ms current / hoisted). The initial CPU baseline also measured about
570 ms for that API. Both builds use O3 and the same OpenBLAS library, and installed
cppdlr headers are identical. The cause of the API discrepancy was not isolated;
do not combine the older CPU-prefix baseline with CUDA timings to inflate speedups.
The table uses the smaller host time from the matched run, and results remain
specific to those build environments and samples.

## Device residency and memory

The N32, mixed-dynamic/reuse and N128 traces audit first and subsequent calls.
All captured traversal ranges have zero allocation/deallocation events and zero
explicit host/device copies. Constructor warmup removes lazy driver kernel allocations
observed in an earlier first-call trace. External operator uploads/resizing occur before
traversal. Work tensors remain device allocations; device-to-device copies persist.

| Trace | Traversal ranges | Nsight-visible peak allocated bytes | Largest UVM migration |
|---|---:|---:|---:|
| [N32/r26](../../../dense_cuda/thinkstationP620_RTXA4000/trace_n32_r26.csv) | 18 | 39,021,389 | 4,096 |
| [Dynamic/reuse](../../../dense_cuda/thinkstationP620_RTXA4000/trace_dynint.csv) | 12 | 29,816,293 | 4,096 |
| [N128/r26](../../../dense_cuda/thinkstationP620_RTXA4000/trace_n128_r26.csv) | 18 | 194,503,565 | 61,440 |

UVM migration is reported explicitly. Pointer tables and solver/library status use
managed memory; large work tensors do not. N128's one rank-time tensor is 6,815,744
bytes, much larger than the first-call metadata migration. The peak includes visible
device/managed/static library allocations, not physical UVM residency or unreported
driver/context caches. N128 application-owned evaluator buffers are 157,931,224 bytes
(150.61 MiB). CSV primitive memory counts refer to the standalone DLR object and
its local tensors/operator, excluding the primary evaluator still alive in that
process; they are not process peaks. API memory counts describe the primary evaluator.

NVTX separates vertex, special vertex, LU solves, convolution, reflection, external
products and non-conjugating traces. [Phase detail](../../../dense_cuda/thinkstationP620_RTXA4000/trace_phases_detail.csv) records
each occurrence contained in traversal; [its traversal audit](../../../dense_cuda/thinkstationP620_RTXA4000/trace_phases_traversal.csv)
retains the allocation/copy gate. CUDA-event primitive times include gaps caused by
synchronous adapters; Nsight kernel times sum active kernels. In the original N32
trace, combined rank/batched GEMMs account for about 65% of active kernel time and
solves about 29%. This supports matrix products as a major cost, but does not prove
batched GEMMs alone dominate wall time. Profiling times are not speedup measurements.
See [README](../../../dense_cuda/thinkstationP620_RTXA4000/ORIGINAL_README.md) for reproduction and trace limitations.

## Acceptance audit

| Requirement | Evidence |
|---|---|
| Separate guarded evaluator, inherited CUDA support, CPU compatibility | `dense/cuda/{diagram_evaluator,device_ops,dlr_ops}`; both Release builds and all 33 C++ CTests pass |
| Complex layouts, disjoint batched outputs, repeated operators, weighting, pack/trace without conjugation | Five actual-GPU DeviceOps tests, including complex trace/layout comparisons |
| LU solve orientation/pivots, reconstruction, reflection, noncommuting convolution and beta | Four actual-GPU DlrOps tests; representative ranks, symmetrized/unsymmetrized meshes and cancellation cases |
| Constructors, all/single/vector/count SE and pairs, signs, both directions, multiplicity | Existing analytic one/two-fermion seams through order three, unitary rotation, `dense/dynint_parity` |
| Dynamic interactions, real/complex basis, bosonic/fermionic kernels | Spin-rotation ladder/Cartesian tests including imaginary Sy; parity and constructor checks |
| Right traversal, reflection, SPGF flavours, one-time operators/statistics | Analytic/rotation SPGF tests; correlator statistics, explicit helper flags and mixed-statistics rejection |
| Long indices/counts and validation before narrowing | `flat_index_range`: counts beyond INT_MAX, order-six selected calls; bad index/prefix checks |
| Current input upload, noncontiguous packing, output shape, reuse after errors, empty endpoints/subsets | `propagator_layout` and dynamic reuse seams; actual strided host view and changed data/operator sets |
| RAII, synchronization/status safety, error context and checked memory arithmetic | Owned native handle/pointer tables, persistent scratch, checked size/byte/BLAS dimensions; NDA allocation errors name cudaMalloc and CUDA error |
| Device residency, inner allocation freedom and observable peak memory | Four traversal trace summaries above; auditor also rejected the pre-warm allocation trace |
| Measured primitives, special vertex, traces, full API and rank/size regimes | Benchmark CSVs, CPU-prefix control and phase trace; transform-hoisting comparison |
| GPU execution and memory checking | CUDA C++ suite executes GPU cases; focused DeviceOps/DlrOps/evaluator Compute Sanitizer checks report zero errors |
| Persistent recovery and separable commits | Supervisor 23 tests, simulated quota reset/races, detached ts service/boot registration; separate plan/log commits on dev_cuda, no push |

Most existing numerical bounds remain unchanged. The order-six selected long-index
CUDA comparison uses 3e-12 relative: 20 DLR solves amplify small rounding differences;
12 nearby indices measured roughly 1.4e-13–1.3e-12 relative and below 4.4e-20 absolute.
CPU/reference assertions retain 1e-13. This exception is limited to that stress case;
it does not justify changing tolerances elsewhere.

No real quota exhaustion or reboot was deliberately induced; recovery was tested
with the actual local transport and simulated backend state, plus live read-only
checks. Allocation-failure context was verified from NDA source rather than exhausting
the GPU. Python bindings/solver integration, block-sparse CUDA, asynchronous scheduling,
custom fusion, cross-backbone batching and multi-GPU support remain outside scope.

### Diagnostic phase totals

N32/r26, four selected indices (NCA two), orders one through three, first and
subsequent ordinary/pairs/SPGF calls, under Nsight. Totals cover all 18 traversals;
phases are disjoint and exclude input/output boundaries. Wall times include host
scheduling and synchronization; summed kernels do not.

| Phase | Occurrences | Wall ms | Active kernel ms |
|---|---:|---:|---:|
| convolution | 144 | 77.06 | 55.18 |
| endpoint_products | 90 | 24.96 | 18.30 |
| reflection | 18 | 1.31 | 0.96 |
| special_vertex | 60 | 123.59 | 64.29 |
| trace | 90 | 22.31 | 8.70 |
| vals2coefs | 288 | 89.40 | 67.65 |
| vertex | 144 | 18.77 | 13.63 |

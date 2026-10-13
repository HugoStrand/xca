# N4 optimization baseline — 2026-10-11

Evidence after reorganization: [CPU/shared baseline](../../dense_cpu/thinkstationP620_RTXA4000/baseline/), [native and larger-dimension runs](../../dense_cuda_small/thinkstationP620_RTXA4000/results/), [Solver workload](solver-workload-current/). Paths in historical commands below refer to the recorded revision.

Final contract dense-small-v2, benchmark code c1a87d7, production source064b64e
(23 protected files unchanged). RTX A4000, Threadripper PRO3945WX, clang19 Release,
CUDA13.4, one OpenMP/OpenBLAS thread. Seven serial runs are preserved under baseline/:
three quick repeats, throughput, method/order coverage, rank26 control and CPU-prefix
control. Each has raw samples, summary and manifest. All224 backend/case groups pass;
GPU results exactly match the unchanged CUDA reference in these runs. Methods include
se/pairs/spgf through order12 with three sampled indices; no exhaustive enumeration.

## Latency and throughput

Median of three process medians, rank16, crossing topology, ordinary self-energy:

| Order | Selected indices | CPU ms | CUDA ms | CPU/CUDA |
|---|---:|---:|---:|---:|
| 1 | 1 | 0.0227 | 0.2497 | 0.091 |
| 1 | 2 (all) | 0.0401 | 0.4597 | 0.087 |
| 3 | 1 | 0.3149 | 1.9523 | 0.161 |
| 3 | 8 | 2.4887 | 15.0403 | 0.165 |
| 6 | 1 | 0.7477 | 4.4271 | 0.169 |
| 6 | 8 | 5.9463 | 35.5657 | 0.167 |
| 8 | 1 | 1.0332 | 6.2690 | 0.165 |
| 8 | 8 | 8.3149 | 49.7882 | 0.167 |

One-index order6 GPU process medians span4.407–4.438ms; order8 spans6.236–6.293ms.
This baseline is about6x slower than CPU at these higher orders. Vector submission
still traverses indices serially: at128 selected indices, order6 takes599.06ms CUDA
versus96.47ms CPU, order8 takes831.04ms versus133.12ms. Per-index cost does not
improve materially. Requested index count is a workload size, not a parallel batch
capacity in the baseline. Different counts use different deterministic index samples.

CPU-only-prefix order8/B1 measured1.016ms versus about1.033ms for host evaluation in
the CUDA-enabled executable. The previous N32 prefix discrepancy is not evident here.
Use matched manifests when comparing experiments; these finite samples are not
confidence intervals. Quick runs took about4s each; throughput about61s; coverage
about8s on this machine, including untimed checks. Setup is charged per fresh timed
object; first calls include lazy schedule/graph preparation. Context is already warm.

## Profiling evidence

N4/r16, order6, crossing, one index, three measured CUDA calls (first, warmup, sample):

- 257 GPU kernels per traversal; about335 cudaDeviceSynchronize calls per complete API
call, including benchmark completion boundaries. Other runtime calls are in
`baseline/trace_n4_runtime.csv`.
- Traversal wall time under Nsight averages5.53ms; summed active kernels2.814ms,
including1.348ms GEMMs and1.251ms solve-family kernels. These are diagnostic profiled
times, not the unprofiled baseline used for speed ratios.
- 20 values-to-coefficients conversions and10 convolutions per traversal. Across the
three calls, conversion phases consume5.860ms wall/4.075ms active kernels;
convolution7.905/3.428ms; vertices1.429/.612ms; special vertices.791/.279ms.
- No allocation/deallocation or explicit host/device copies inside traversal;53,248
bytes of device-to-device copies per traversal. No UVM migrations in those ranges.
- Complete API ranges show two cudaMalloc/cudaFree calls per call outside traversal.
Identify their ownership before choosing reusable upload/result staging.
The observed trace peak38,466,613 allocated bytes includes both reference/candidate
objects and library allocations, not physical residency or hidden context caches.

Reproduce after sourcing the CUDA environment:

```bash
nsys profile --trace=cuda,nvtx,cublas,cusolver --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
  --output=cbuild-small-profile/n4-r16-order6 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cbuild-cuda/examples/dense_cuda_small_benchmark \
  --orders 6 --topologies crossing --indices 1 --samples 1
nsys export --type=sqlite --output=cbuild-small-profile/n4-r16-order6.sqlite \
  cbuild-small-profile/n4-r16-order6.nsys-rep
python3 tools/analyze_dense_cuda_trace.py cbuild-small-profile/n4-r16-order6.sqlite \
  --within bench.cuda.sample
# Add --phases for the phase CSV; constructor/validation ranges are excluded.
```

The runtime CSV counts CUPTI runtime events whose start lies in each NVTX range
named bench.cuda.sample, joining nameId to StringIds. The three ranges include
boundary completion checks, unlike the inner traversal kernel/allocation audit.
Graphs may remove submission overhead, but cannot alone increase the useful work
in a tiny kernel. Index batching also widens DLR rank transforms and shared-LU RHS
solves, which are substantial costs here; fused operations can reduce launch count
and scratch traffic. These measurements motivate the ordered experiments in the plan.

## Numerical scope and validation

The benchmark requires relative parity with the unchanged CUDA evaluator even for
very small terms. CPU agreement is additionally required through order3. Higher-order
CPU differences are recorded, not hidden behind a fixed absolute tolerance or claimed
as new accuracy validation. One order8 probe had norm1.18e-18 and absolute difference
4.85e-25 (relative4.10e-7). Coverage found a selected order10 crossing lane with relative
CPU discrepancy1.35; all individual absolute differences in that case were at most
3.54e-34, while the summed CPU result norm was4.46e-28. Max per-index absolute and
relative columns may refer to different lanes. Rank26's worst sampled per-index CPU
relative discrepancy was3.37e-9 in these selected cases. Results therefore depend
strongly on conditioning/cancellation and rank. Preserve the baseline but investigate
numerical accuracy before relying on such tiny higher-order terms scientifically.

Both CPU/CUDA benchmark builds pass the three focused contract tests (zeroed tiny
result rejection, nonfinite rejection, long-index selection and malformed input checks).
CLI invalid-input and comparison-mismatch tests passed. Empty/duplicate subsets and
changed propagator data are checked on actual GPU for each case. The supervisor's
24 tests pass. Production code remains untouched; the previous port's analytic suites
remain the additional acceptance gate for later experimental implementations.

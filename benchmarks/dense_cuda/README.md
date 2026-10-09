# Dense evaluator baseline

The completed port's measurements and acceptance audit are in [results.md](results.md).
The initial baseline below is retained as historical evidence.

`cpu_baseline.csv` records the initial CPU measurements on ts on 2026-10-10.
These are bounded synthetic C++ workloads, not solver benchmarks or GPU timings.

Machine: AMD Ryzen Threadripper PRO 3945WX (12 cores). Release build, clang-19,
CPU TRIQS 4.0 prefix. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
GPU available for the port: NVIDIA RTX A4000, 16 GB; no CUDA speedup is claimed here.

Workload: beta 2, DLR cutoff 40, tolerance 1e-12, rank 26, N in 2/4/16/32/64,
log2(N) occupation-basis fermion operators, two hybridization poles and complex
coefficients. The propagator is a dense Fourier rotation of a normalized diagonal
imaginary-time propagator. Orders 1–3 use evenly spaced flat-index subsets of at
most eight entries; NCA has two entries and includes its complete special vertex.
Construction is timed separately from host workload generation. First API calls
and average times over three subsequent calls are separate columns.

Reproduce:

```bash
source ~/env/xca-cpu.sh
(cd cbuild && bash ../cmake.sh)
cmake --build cbuild --target dense_cuda_benchmark --parallel 12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/examples/dense_cuda_benchmark 32 8 3
```

Arguments are N, maximum selected indices, and warm repeats. N supports powers of
two through 128. The benchmark records whole self-energy API time and isolated
left-vertex, values-to-coefficients, time-ordered convolution and reflection time.
It does not derive exact phase fractions of a whole evaluation. Initial construction
and operation initialization make the first-call times noisy; these are one-run
baselines and not statistical confidence intervals.

The `convolution_hoisted_cpu` row performs the same convolution with the two rank
transforms computed once each outside the time-node loop. Its `checksum` column is
maximum absolute difference from the current cppdlr result (zero for these runs);
other operation checksums are maximum absolute result entries. The benchmark rejects
hoisted differences larger than 1e-12.

At N=32, the warm convolution time is 7.12 ms in current cppdlr and 1.77 ms with
hoisted transforms; a left vertex is 0.157 ms. At N=64, corresponding times are
30.73 ms, 10.56 ms and 1.04 ms. This demonstrates that repeated rank-transform work
has measurable cost in this workload. It is not evidence that CUDA has accelerated
anything. Final port measurements must distinguish this CPU algorithm improvement
from device offload and include construction, transfers, memory and residency traces.

## CUDA comparison and residency audit

The extended benchmark accepts an optional fourth argument (DLR cutoff, default
40) and fifth argument `api-only` to omit standalone CUDA primitive measurements.
The CPU rows in a CUDA-enabled executable still execute the host dense evaluator;
the two backends receive identical propagators, poles, topology and index vectors.
All returned CUDA results are checked against CPU at maximum absolute error 1e-12;
non-finite differences fail. By-pairs and single-particle API rows are also measured.
CPU-only compilation retains its host measurements.

New CSV columns append cutoff, warm default-stream CUDA-event time, application-owned
device/unified buffer bytes and maximum absolute comparison error. Event timing is
for isolated CUDA primitives and includes gaps introduced by their synchronous host
adapters; it is not the sum of active kernel times. API timings are synchronized wall
time including uploads, host scheduling and copy-back. Event time zero for API rows
means not separately measured. Values-to-coefficients primitive error is reconstruction
error after device coefs2vals, rather than direct ill-conditioned coefficient error.
Construction includes lazy module/kernel warmup; context initialization is separate.
First means the first measured call of that workload within the process, after any
earlier phases. Warm time averages the next three calls. These are single-run samples,
not confidence intervals. The initial pre-port CPU CSV uses the older column schema.

Residency instrumentation uses header-only NVTX ranges around uploads, downloads and
flat-index traversal. Constructor warmup primes batched/rank GEMM, weighting and DLR
paths with existing scratch. A first-call trace
without standalone primitive prewarming originally found two 16-byte driver allocations
inside cuKernelGetFunction; warming removes them before traversal.

Verified trace summaries:

- `trace_n32_r26.csv`: N=32, rank 26, two poles, four selected indices (NCA two),
  orders 1–3, first and subsequent ordinary/pairs/single-particle calls: 18 traversal
  ranges, **zero allocation/deallocation events and zero explicit host/device copies**.
- `trace_dynint.csv`: existing mixed-dynamic-interface/reuse test, including density
  correlators and failed-prefix recovery: 12 ranges with the same zero-event result.
- `trace_n128_r26.csv`: N128 first/warm API-only calls, 18 ranges with the same
  zero-event result; visible peak 194,503,565 bytes and largest UVM migration 61,440 bytes.
- Unified-memory migrations remain visible: maxima 4096 bytes per migration in N32/dynamic traces, consistent
  with pointer-table/status pages. Large work-tensor copies are device-to-device.
  Host/unified copies are reported separately, rather than treating metadata migration
  as absent. CUDA allocations include driver-visible memory events, not just malloc APIs.
- Nsight-visible peak device/managed/static allocations are 39,021,389 bytes for the N32
  trace and 29,816,293 for the dynamic test. This includes observable library allocations;
  it does not claim physical UVM residency or unreported context/driver cache memory.
  Application-owned bytes are reported separately by the benchmark.

Profiler: Nsight Systems 2026.3.2.476; GPU RTX A4000, driver 580.178.04, 16376 MiB.
The traces enable CUDA runtime/driver activity, memory usage and UVM page faults,
with CPU sampling/context switches disabled. Profiling times are diagnostic and must
not be used as the unprofiled speed measurements. Full `.nsys-rep` and SQLite exports
are machine-local under ignored `cbuild-profile/`; tracked CSVs preserve the audit.

Reproduce the N32 trace:

```bash
source ~/env/xca-cuda.sh
cmake --build cbuild-cuda --target dense_cuda_benchmark --parallel 12
mkdir -p cbuild-profile
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nsys profile \
  --trace=cuda,nvtx,cublas,cusolver --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true --force-overwrite=true \
  --output=cbuild-profile/dense-n32-r26 \
  cbuild-cuda/examples/dense_cuda_benchmark 32 4 1 40 api-only
nsys export --type=sqlite --force-overwrite=true \
  --output=cbuild-profile/dense-n32-r26.sqlite cbuild-profile/dense-n32-r26.nsys-rep
python3 tools/analyze_dense_cuda_trace.py cbuild-profile/dense-n32-r26.sqlite
```

The audit returns nonzero for any inner allocation or explicit host/device copy,
or incomplete/unknown trace data. Its GEMM/solve timings classify demangled kernel
families (`gemm`/split-K reduction versus TRSM/pivot/status kernels); remaining kernels
include GEAM, DGMM and vector operations. Combined GEMMs account for about 65% of
active kernel time in the N32 trace, solves about 29%; these are not whole-call shares,
nor a proof that batched GEMMs alone dominate. Launch and synchronization overhead
must be judged from wall time alongside the active kernel sum.

Final diagnostic ranges also separate vertices, special vertices, LU solves,
convolution, reflection, endpoint products and traces. Profile the same command
with output `dense-phases-n32-r26`; export its SQLite and pass `--phases` to the
auditor for `trace_phases_detail.csv`. Without that flag it emits the traversal audit.
Constructor warmup phases are excluded from the phase report.

`owned_device_bytes` on primitive rows covers the standalone DLR object and local
buffers, excluding the primary evaluator still alive. API rows cover that evaluator.
These counts exclude context/library caches and must not be read as process peaks.

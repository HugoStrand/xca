# Dense cuBLAS benchmarks

The bounded fixture entry point measures the cuBLAS evaluator against CPU using
[shared dense workloads and timing/CSV semantics](../dense_common/README.md).
Its CUDA reference factory, synchronization and scoped profiling hooks live in
`adapter.hpp`. Configure with `Build_Benchmarks=ON` and use the runtime environment
matching the selected TRIQS prefix.

```bash
cmake --build cbuild-cuda --target dense_cuda_fixture_benchmark --parallel 6
python3 benchmarks/dense_cuda/run.py --output results/my_machine/cublas/run-01 -- \
  --orders 1,3,6,8 --indices 1,8 --samples 3
```

The runner defaults to `cbuild-cuda/benchmarks/dense_cuda/dense_cuda_fixture_benchmark`;
use `--binary` for another build. CPU builds can run this entry point with `--backend cpu`.

## Primitive and larger-matrix workload

`dense_cuda_benchmark` measures whole self-energy, paired and single-particle APIs,
plus vertices, DLR conversion, convolution and reflection. Its synthetic workload
uses beta=2, tolerance=1e-12, two poles, complex coefficients, occupation-basis
operators and a Fourier-rotated normalized propagator. Orders one through three
use evenly spaced index subsets. CUDA results are checked against CPU at maximum
absolute error 1e-12, with non-finite comparisons rejected.

```bash
cmake --build cbuild-cuda --target dense_cuda_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild-cuda/benchmarks/dense_cuda/dense_cuda_benchmark 32 8 3 40 api-only > measurements.csv
```

Arguments are dimension (powers of two through 128), maximum selected indices,
warm repeats, optional DLR cutoff (default 40), and optional `api-only` to omit
standalone CUDA primitives. CPU builds retain host measurements. Construction,
context, first calls and mean subsequent times are separate rows. First calls may
follow primitive warmup in the same process; record whether `api-only` was used.

The `convolution_hoisted_cpu` row performs the same convolution with two rank
transforms computed outside the time-node loop. Its checksum is maximum absolute
difference from cppdlr, rejected above 1e-12; other host primitive checksums are
result norms. CUDA conversion error measures reconstruction after `coefs2vals`,
rather than direct ill-conditioned coefficient agreement.

CUDA-event times apply to isolated primitives and include synchronous-adapter gaps;
zero on API rows means not separately measured. API wall times include transfers,
host scheduling, owning results and completion. Primitive workspace counts cover
the standalone object/local buffers; API counts cover the evaluator. Neither
includes context/library caches or represents peak physical residency.

## Profiling

NVTX ranges identify uploads/downloads, traversal and arithmetic phases. Capture
CUDA memory and unified-memory faults to distinguish explicit transfers from UVM
migration. Profiled durations are diagnostic, not unprofiled speedup measurements.

```bash
mkdir -p results
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nsys profile \
  --trace=cuda,nvtx,cublas,cusolver --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
  --output=results/dense-n32-r26 \
  cbuild-cuda/benchmarks/dense_cuda/dense_cuda_benchmark 32 4 1 40 api-only
nsys export --type=sqlite --output=results/dense-n32-r26.sqlite results/dense-n32-r26.nsys-rep
python3 tools/analyze_dense_cuda_trace.py results/dense-n32-r26.sqlite
```

The auditor rejects inner allocations, explicit host/device copies and incomplete
trace data. Add `--phases` for disjoint diagnostic phase totals; constructor warmup
is excluded. Summed active kernels do not include host scheduling/synchronization.

[Workstation profiling](thinkstationP620_RTXA4000/README.md) and
[matched comparison reports](../dense_common/thinkstationP620_RTXA4000/README.md)
provide RTX A4000 evidence. The target is GH200-class hardware; repeat measurements
before choosing dispatch thresholds elsewhere.

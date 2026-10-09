# Dense evaluator baseline

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

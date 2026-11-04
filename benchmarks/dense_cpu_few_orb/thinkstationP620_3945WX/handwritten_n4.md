# ThinkStation P620 / Threadripper PRO 3945WX — handwritten N=4

This records the earlier handwritten-kernel implementation. For the current
Eigen-based dimension sweep, see the [main report](README.md). Dependency and
dispatch statements below describe the recorded source revisions.

Measured on 2026-11-04 using the AMD Ryzen Threadripper PRO 3945WX (12 cores,
24 hardware threads), Clang 19.1.1, Release `-O3 -DNDEBUG -std=c++23`, and system
OpenBLAS. No `-march=native` or fast-math flags were used. Each process was pinned
to CPU 2, with `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; no MPI launcher or
GPU. The `schedutil` governor and frequency boost remained enabled. The machine
was not isolated from unrelated activity, but builds, tests and benchmarks ran
sequentially.

Evaluator source: XCA `14de81f` (original buffer reuse) and `8e8e4b3` (FewOrb).
Installed dependencies: TRIQS `2930120`, NDA `4e59077`, cppdlr `b9b24ab`.
The original baseline already includes fast convolution **and this round's
reusable product-buffer changes**. It is therefore a stronger baseline than the
older dense batching tables. These results do not measure the old cubic-time
convolution or a complete impurity-solver iteration.

Benchmark source and raw results: XCA `5132144`.

Raw results: [evaluator timings](timings.csv), [Eigen comparison](eigen_gemm.csv).
The [generic README](../README.md) defines the model, cases and timing protocol.
Times below are median **milliseconds per call**, five trials with rotated
variant order. Cache numerical preparation is included in every call; workspace
and cache allocations have been warmed.

## Complete topology calls

These are the primary comparison: every variant uses the same topology-level
API boundary, including backbone preparation and output wrapping. Order two
contains 32 ordinary or 16 paired/correlator backbones; order three contains 512
ordinary or 256 paired/correlator backbones. Pair entries evaluate both endpoint
directions, so compare variants within a row.

### DLR rank 13

| Evaluation | Order | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 2 | 32 | 1.615 | 1.290 | 1.084 | 1.49× | 1.19× |
| Paired self-energy | 2 | 16 | 1.109 | 0.879 | 0.778 | 1.42× | 1.13× |
| Correlator | 2 | 16 | 0.671 | 0.534 | 0.435 | 1.54× | 1.23× |
| Self-energy | 3 | 512 | 41.484 | 33.195 | 26.131 | 1.59× | 1.27× |
| Paired self-energy | 3 | 256 | 25.434 | 20.280 | 16.767 | 1.52× | 1.21× |
| Correlator | 3 | 256 | 18.568 | 14.832 | 11.337 | 1.64× | 1.31× |

### DLR rank 103

| Evaluation | Order | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 2 | 32 | 21.287 | 14.691 | 11.568 | 1.84× | 1.27× |
| Paired self-energy | 2 | 16 | 12.963 | 9.150 | 7.669 | 1.69× | 1.19× |
| Correlator | 2 | 16 | 10.021 | 6.886 | 5.434 | 1.84× | 1.27× |
| Self-energy | 3 | 512 | 603.779 | 408.179 | 303.721 | 1.99× | 1.34× |
| Paired self-energy | 3 | 256 | 338.446 | 232.522 | 180.369 | 1.88× | 1.29× |
| Correlator | 3 | 256 | 291.339 | 196.160 | 144.516 | 2.02× | 1.36× |

At rank 103, the uncached implementation improves the original by
**1.42–1.49×**. Edge reuse adds **1.19–1.36×**, for a total **1.69–2.02×**.
At rank 13 the total gain is **1.42–1.64×**. Every complete-topology median
improves with both steps. The effects are workload-dependent rather than one
universal speedup factor.

## Bounded order-five samples

Each call sums 63 stratified prepared backbones. These include numerical work
and returned-result allocation, but exclude backbone preparation. The dense
baseline returns one array per backbone; FewOrb returns one span sum. These
dense-to-FewOrb ratios combine arithmetic and that API allocation difference;
they do not isolate caching or predict a complete order-five topology time.
The cache-gain column compares the two FewOrb variants at the same API boundary
and isolates edge reuse. Order-two and order-three prepared rows are also
available in the CSV.

### DLR rank 13

| Evaluation | Order | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 5 | 63 | 8.937 | 7.153 | 5.460 | 1.64× | 1.31× |
| Paired self-energy | 5 | 63 | 10.110 | 8.037 | 6.348 | 1.59× | 1.27× |
| Correlator | 5 | 63 | 8.392 | 6.700 | 5.029 | 1.67× | 1.33× |

### DLR rank 103

| Evaluation | Order | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 5 | 63 | 139.668 | 92.664 | 67.849 | 2.06× | 1.37× |
| Paired self-energy | 5 | 63 | 148.492 | 100.130 | 75.146 | 1.98× | 1.33× |
| Correlator | 5 | 63 | 136.678 | 91.039 | 66.263 | 2.06× | 1.37× |

At rank 103 the sampled order-five gain is **1.98–2.06×**; reuse alone contributes
**1.33–1.37×** over uncached FewOrb. The cache saves the repeated coefficient
solve and Hilbert transform of the fixed edge operand. The running backbone
operand and transforms back to nodal values still have to be computed, which
limits the benefit even with very high hit rates.

## Cache use and accuracy

Complete topology hit rates are **90.6–99.7%**, using just three distinct edge
recipes at order two and seven at order three. Order-five samples reach
**96.0–96.4%**, with 18 or 20 recipes. The maximum active numerical cache payload
is **1,054,720 bytes (1.006 MiB)**, below the default 8 MiB budget. At rank 103,
one entry occupies 52,736 bytes for its two complex `(103,4,4)` arrays.

CSV byte counts describe entries prepared in that call. Retained allocations can
reflect a larger earlier call; metadata, fixed scratch, real DLR tables and
Action storage are additional. Allocation slots remain bounded by the configured
numerical budget. This model never exhausts it during benchmarking; budget
exhaustion and uncached fallback are covered by correctness tests instead.

All 90 evaluator rows passed finite/nonzero-reference checks. The largest
relative maximum discrepancy is **8.16e-14**, for the rank-103 order-five
correlator sample; the threshold is 2e-10. Cached and uncached FewOrb have
identical displayed discrepancies in every case. The complete rebuilt CPU suite
also passed **88/88** CTest entries, including serial/three-rank MPI checks,
cache limits, same-address propagator mutation, complex retarded interactions,
strided inputs, N=2/N=8 fallback, and prepared-input recovery.

## Eigen kernel experiment

Headers: [Eigen 5.0.1 release archive](https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz),
SHA256 `e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec`.
Headers were unpacked into the ignored build tree; no Eigen dependency was added
to production XCA. Eigen and handwritten code used the generic compiler target;
OpenBLAS selects its own CPU-specific kernels at runtime.

Median **nanoseconds per complex 4×4 product**, over stacks of 103 matrices and
1024 repetitions. Unit-alpha cases compile with alpha=1 to match evaluator use;
general complex-alpha cases are separate controls. All kernels use independent
input/output storage. Correctness is checked against a long-double reference.

| Operation | Handwritten | Eigen fixed 4×4 | NDA/OpenBLAS | Eigen vs handwritten |
| --- | ---: | ---: | ---: | ---: |
| C = AB | 83.7 | 53.6 | 128.3 | 1.56× |
| C += AB | 98.3 | 57.3 | 116.6 | 1.72× |
| C = αAB, complex α | 83.3 | 60.5 | 128.5 | 1.38× |
| C += αAB, complex α | 97.9 | 63.7 | 115.9 | 1.54× |

The largest relative kernel discrepancy is **2.59e-16**. Eigen improves on the
handwritten unit-alpha products by **1.56–1.72×** in this experiment. The
production evaluator retains handwritten loops as requested. This microbenchmark
supports a future evaluator-level Eigen comparison, but does not establish its
end-to-end speedup: DLR transforms and solves, endpoint sums and cache lookup
remain outside the small product timing.

## Limits and next measurements

These measurements support the separate FewOrb implementation for this N=4
workload. They do not establish a dimension crossover, a solver-level gain, or a
performance guarantee across other CPUs. The fixture has only two fitted poles;
more diverse edge recipes could reduce reuse and reach the cache budget sooner.
The model/Action, evaluator construction, and first-use cache allocation are not
timed. One low-rank prepared pair case had large maximum-time outliers, although
its median improved; raw minimum/maximum timings are retained.

Useful next comparisons are more fitted poles and mixed topology collections,
smaller cache budgets, realistic propagators from solver iterations, a full
solver iteration, and the optional Eigen product inside the same evaluator.
A dimension sweep at N=2,8,16 is needed before selecting further fixed-size
kernels or an automatic crossover. No GPU conclusion follows from these CPU
results.

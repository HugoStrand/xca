# ThinkStation P620 / Threadripper PRO 3945WX

Measured on 2026-11-04. The Eigen-based FewOrb evaluator improves the original
at N=2 and N=4. N=8 is workload-dependent, and N=16 is generally slower for
complete order-two calls (the rank-103 correlator is near parity). The kernel comparison shows that OpenBLAS overtakes the current
Eigen products at N=8 on this machine. Fixed-size specialization alone therefore
does not justify using Eigen at every small dimension.

FewOrb now compiles Eigen products for N=2,4,8,16, retaining handwritten products
as `small_gemm_handwritten<N>` and BLAS fallback for other dimensions. The
requested Eigen default is used throughout this sweep; dimension-dependent BLAS
selection remains a follow-up experiment.

## Setup and scope

AMD Ryzen Threadripper PRO 3945WX (12 cores, 24 hardware threads), Clang 19.1.1,
Release `-O3 -DNDEBUG -std=c++23`, Eigen 5.0.1 and system OpenBLAS. No
`-march=native` or fast-math flags. Each process was pinned to CPU 2 with
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`, without MPI or GPU. The
`schedutil` governor and frequency boost remained enabled. Builds, tests and
timings ran sequentially; the machine was not isolated from unrelated activity.
Eigen and handwritten code use the generic compiler target, while OpenBLAS
selects CPU-specific kernels at runtime. The measured crossover applies to this
build and machine.

Evaluator source: XCA `cc4f66f`; benchmark source and raw results:
`b46b4dd`. Dependencies: TRIQS `2930120`, NDA `4e59077`, cppdlr `b9b24ab`.
The baseline already includes fast convolution and reusable GEMM destinations.
Eigen is a normal public dependency, using the 5.0.1 source release in this run
(archive SHA256
`e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec`).

Raw results: [evaluator sweep](timings_eigen_dimensions.csv) and
[kernel sweep](eigen_gemm_dimensions.csv). The [generic README](../README.md)
describes the fixture and reproduction commands. The earlier handwritten N=4
measurements remain in [their report](handwritten_n4.md), [evaluator CSV](timings.csv)
and [isolated Eigen experiment](eigen_gemm.csv).

N is Hilbert-space dimension, with N=2,4,8,16 corresponding to one through four
fermion modes. The N=4 fixture is unchanged. Increasing N also increases the
number of arc flavours and topology work; comparisons across dimensions are not
pure matrix-size scaling. Compare variants within a row.

Times below are median **milliseconds per call**, over five rotated trials.
Numerical cache preparation is included anew in each call; allocation is warmed.
The default cache budget remains 8 MiB for all dimensions. These are diagram
evaluator measurements, not full solver iterations.

## Complete order-two topology calls

These include backbone preparation and matching public-call/result-wrapper
boundaries. Each paired backbone accounts for both distinguished-arc directions.
The speedup column is dense time divided by FewOrb time: values below one mean
FewOrb is slower. Cache gain compares uncached and cached FewOrb at the same API
boundary.

### DLR rank 13

| N | Evaluation | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 16 | 0.337 | 0.184 | 0.145 | 2.33× | 1.27× |
| 2 | Paired self-energy | 8 | 0.208 | 0.107 | 0.089 | 2.34× | 1.20× |
| 2 | Correlator | 8 | 0.155 | 0.091 | 0.073 | 2.13× | 1.25× |
| 4 | Self-energy | 32 | 1.614 | 1.230 | 1.023 | 1.58× | 1.20× |
| 4 | Paired self-energy | 16 | 1.107 | 0.834 | 0.734 | 1.51× | 1.14× |
| 4 | Correlator | 16 | 0.670 | 0.508 | 0.410 | 1.63× | 1.24× |
| 8 | Self-energy | 48 | 7.783 | 11.096 | 10.032 | 0.78× | 1.11× |
| 8 | Paired self-energy | 24 | 5.670 | 8.426 | 7.904 | 0.72× | 1.07× |
| 8 | Correlator | 24 | 2.961 | 4.064 | 3.546 | 0.84× | 1.15× |
| 16 | Self-energy | 64 | 50.440 | 87.200 | 81.386 | 0.62× | 1.07× |
| 16 | Paired self-energy | 32 | 39.278 | 70.261 | 67.485 | 0.58× | 1.04× |
| 16 | Correlator | 32 | 17.487 | 29.506 | 26.614 | 0.66× | 1.11× |

### DLR rank 103

| N | Evaluation | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 16 | 4.033 | 2.117 | 1.616 | 2.50× | 1.31× |
| 2 | Paired self-energy | 8 | 2.320 | 1.175 | 0.949 | 2.44× | 1.24× |
| 2 | Correlator | 8 | 2.023 | 1.158 | 0.933 | 2.17× | 1.24× |
| 4 | Self-energy | 32 | 21.362 | 14.123 | 11.081 | 1.93× | 1.27× |
| 4 | Paired self-energy | 16 | 12.986 | 8.755 | 7.306 | 1.78× | 1.20× |
| 4 | Correlator | 16 | 10.015 | 6.667 | 5.199 | 1.93× | 1.28× |
| 8 | Self-energy | 48 | 108.717 | 112.982 | 96.084 | 1.13× | 1.18× |
| 8 | Paired self-energy | 24 | 68.369 | 79.180 | 70.962 | 0.96× | 1.12× |
| 8 | Correlator | 24 | 48.725 | 46.499 | 38.273 | 1.27× | 1.21× |
| 16 | Self-energy | 64 | 650.480 | 848.070 | 748.990 | 0.87× | 1.13× |
| 16 | Paired self-energy | 32 | 439.567 | 637.809 | 589.861 | 0.75× | 1.08× |
| 16 | Correlator | 32 | 271.819 | 319.527 | 272.857 | 1.00× | 1.17× |

At N=2 the gain is 2.13–2.50×; at N=4 it is 1.51–1.93×. N=8 regresses at
rank 13 and gives mixed results at rank 103. N=16 generally regresses despite
cache reuse; its rank-103 correlator is near parity (0.38% slower median). The
low-rank regressions occur while the cache has
ample capacity, so increasing its budget cannot remove the underlying matrix
product cost.

## Complete order-three topology calls

The default sweep includes complete order three for N=2 and N=4. For N=8 and
N=16 it uses bounded prepared samples at orders three and five; complete order
three is available with `--full-order3` but was not run here.

### DLR rank 13

| N | Evaluation | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 128 | 4.674 | 2.652 | 1.921 | 2.43× | 1.38× |
| 2 | Paired self-energy | 64 | 2.647 | 1.437 | 1.083 | 2.44× | 1.33× |
| 2 | Correlator | 64 | 2.223 | 1.301 | 0.956 | 2.33× | 1.36× |
| 4 | Self-energy | 512 | 41.430 | 31.769 | 24.761 | 1.67× | 1.28× |
| 4 | Paired self-energy | 256 | 25.452 | 19.374 | 15.855 | 1.61× | 1.22× |
| 4 | Correlator | 256 | 18.477 | 14.163 | 10.661 | 1.73× | 1.33× |

### DLR rank 103

| N | Evaluation | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 128 | 59.764 | 31.841 | 23.011 | 2.60× | 1.38× |
| 2 | Paired self-energy | 64 | 32.363 | 16.828 | 12.467 | 2.60× | 1.35× |
| 2 | Correlator | 64 | 29.686 | 16.802 | 12.416 | 2.39× | 1.35× |
| 4 | Self-energy | 512 | 604.537 | 393.581 | 290.426 | 2.08× | 1.36× |
| 4 | Paired self-energy | 256 | 338.635 | 223.375 | 172.336 | 1.96× | 1.30× |
| 4 | Correlator | 256 | 290.962 | 190.398 | 138.867 | 2.10× | 1.37× |

## Bounded order-five samples

Each call sums 63 stratified prepared backbones, excluding backbone preparation.
Dense returns one array per backbone while FewOrb returns one span sum. These
ratios therefore include that API/allocation difference as well as arithmetic;
they do not predict complete order-five topology timings. Cache gain compares
the two FewOrb variants with matching boundaries. Prepared order-two/three rows
are retained in the CSV.

### DLR rank 13

| N | Evaluation | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 63 | 4.237 | 2.365 | 1.705 | 2.49× | 1.39× |
| 2 | Paired self-energy | 63 | 4.521 | 2.491 | 1.813 | 2.49× | 1.37× |
| 2 | Correlator | 63 | 4.093 | 2.369 | 1.684 | 2.43× | 1.41× |
| 4 | Self-energy | 63 | 8.931 | 6.834 | 5.153 | 1.73× | 1.33× |
| 4 | Paired self-energy | 63 | 10.097 | 7.674 | 5.998 | 1.68× | 1.28× |
| 4 | Correlator | 63 | 8.382 | 6.402 | 4.721 | 1.78× | 1.36× |
| 8 | Self-energy | 63 | 25.924 | 34.748 | 29.081 | 0.89× | 1.19× |
| 8 | Paired self-energy | 63 | 30.602 | 42.254 | 36.594 | 0.84× | 1.15× |
| 8 | Correlator | 63 | 23.475 | 30.932 | 25.321 | 0.93× | 1.22× |
| 16 | Self-energy | 63 | 113.484 | 182.748 | 159.460 | 0.71× | 1.15× |
| 16 | Paired self-energy | 63 | 140.966 | 235.346 | 211.968 | 0.67× | 1.11× |
| 16 | Correlator | 63 | 97.639 | 154.746 | 131.653 | 0.74× | 1.18× |

### DLR rank 103

| N | Evaluation | Backbones | Dense | FewOrb, cache off | FewOrb | vs dense | Cache gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 63 | 55.806 | 30.134 | 21.639 | 2.58× | 1.39× |
| 2 | Paired self-energy | 63 | 58.299 | 31.121 | 22.606 | 2.58× | 1.38× |
| 2 | Correlator | 63 | 55.832 | 31.133 | 22.611 | 2.47× | 1.38× |
| 4 | Self-energy | 63 | 139.632 | 90.044 | 65.032 | 2.15× | 1.38× |
| 4 | Paired self-energy | 63 | 148.033 | 96.499 | 71.698 | 2.06× | 1.35× |
| 4 | Correlator | 63 | 136.317 | 88.569 | 63.658 | 2.14× | 1.39× |
| 8 | Self-energy | 63 | 454.221 | 407.274 | 318.673 | 1.43× | 1.28× |
| 8 | Paired self-energy | 63 | 490.989 | 467.502 | 378.628 | 1.30× | 1.23× |
| 8 | Correlator | 63 | 439.167 | 381.431 | 292.629 | 1.50× | 1.30× |
| 16 | Self-energy | 63 | 1874.048 | 2051.243 | 1733.502 | 1.08× | 1.18× |
| 16 | Paired self-energy | 63 | 2096.502 | 2469.806 | 2149.303 | 0.98× | 1.15× |
| 16 | Correlator | 63 | 1766.098 | 1843.223 | 1522.356 | 1.16× | 1.21× |

The longer traversals give edge reuse more opportunities to amortize preparation,
which can overcome some of the N=8/N=16 matrix-product penalty at rank 103. The
ranking therefore depends on both matrix dimension and DLR rank/workload.

## Kernel comparison

Median **nanoseconds per complex matrix product**, 103 independent matrices,
1024 stack repetitions, five rotated trials. These unit-alpha cases match the
evaluator's overwrite and accumulation operations. General complex-alpha
controls are also recorded in the CSV. The Eigen column calls the production
`small_gemm<N>` helper, including its packed/strided layout check.

| N | Operation | Handwritten | Eigen | NDA/OpenBLAS |
| ---: | --- | ---: | ---: | ---: |
| 2 | C = AB | 23.7 | 11.6 | 82.7 |
| 2 | C += AB | 26.8 | 12.1 | 76.3 |
| 4 | C = AB | 84.2 | 63.8 | 129.2 |
| 4 | C += AB | 98.0 | 67.1 | 116.1 |
| 8 | C = AB | 510.8 | 524.0 | 248.9 |
| 8 | C += AB | 563.9 | 515.3 | 212.1 |
| 16 | C = AB | 3817.7 | 2867.7 | 983.2 |
| 16 | C += AB | 4104.0 | 2833.5 | 862.1 |

Eigen wins at N=2 and N=4. OpenBLAS is about 2.1–2.4× faster at N=8 and
2.9–3.3× faster at N=16 for these two operations. Eigen improves on handwritten
loops at N=16 but still trails BLAS. The previous N=4 microbenchmark called raw
Eigen maps directly; its numbers do not measure precisely the same wrapper.
There is no handwritten-versus-Eigen whole-evaluator A/B in this sweep.

## Cache pressure and numerical checks

Each retained recipe stores two complex `(r,N,N)` tensors. At N=16/r=103, each
entry occupies 843,776 bytes (0.805 MiB): nine fit in 8 MiB. The order-five sample
reaches that limit; unmatched recipes are recomputed without evicting retained
entries. Its per-call cache statistics are:

| Evaluation | Hits | Misses | Hit rate | Entries | Payload (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 411 | 93 | 81.5% | 9 | 7.242 |
| Paired self-energy | 416 | 88 | 82.5% | 9 | 7.242 |
| Correlator | 416 | 88 | 82.5% | 9 | 7.242 |

Three recipes suffice for complete order two; seven suffice for the complete
order-three cases that were run. The N=16/r=103 prepared order-three cases also
fit seven entries, so the larger-dimension slowdown is already visible without
cache pressure. CSV byte counts describe active numerical payload, excluding
metadata and fixed scratch. Retained allocation can reflect a larger previous
call, but remains bounded by the numerical budget.

All **324 evaluator rows** (108 cases, three variants each) passed finite-result,
nonzero-reference and relative-error checks. The largest relative maximum
discrepancy is **2.19e-13**, below the 2e-10 threshold. All **48 kernel rows**
passed the long-double reference check; maximum discrepancy **7.32e-16**.
The full rebuilt CPU suite passed **88/88** CTest entries, including fixed
N=2/4/8/16 kernels and evaluator paths, strided and minimally aligned operands,
complex alpha/beta, cache limits and N=32 BLAS fallback. A standalone Eigen
allocation-guard probe passed packed and strided products through N=16, and an
external C++ consumer built against a temporary installation.

## Recommended next experiment

Retain the FewOrb real DLR arithmetic and edge cache but switch only its N=8 and
N=16 products to BLAS, then rerun the same cases. That would isolate whether a
dimension-based Eigen/BLAS choice recovers the larger-dimension regressions
without losing reuse. The current sweep strongly motivates this experiment,
but does not measure that combined implementation. A native-architecture build
should be a separate comparison before generalizing the crossover across CPUs.

For cache policy, compare the default 8 MiB with a budget large enough for every
encountered recipe at N=16/r=103, then add more poles and realistic solver
propagators. This separates capacity pressure from arithmetic cost. Complete
higher-order topologies, full solver timing and multi-core/MPI scaling remain
outside the present measurements; no GPU conclusion follows from them.

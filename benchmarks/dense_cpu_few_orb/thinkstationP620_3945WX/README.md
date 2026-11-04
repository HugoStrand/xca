# ThinkStation P620 / Threadripper PRO 3945WX

## Shared convolution and selected small products

Measured on 2026-11-04. All three variants now use the same real-valued DLR
transforms and solves from `cppdlr::fast_convolve`. This comparison measures the
remaining benefit of FewOrb's small matrix products and edge reuse over the
updated original dense evaluator.

| Variant in CSV | Matrix products | Edge reuse |
| --- | --- | --- |
| `dense` | BLAS at every dimension | Off |
| `few_orb_uncached` | Eigen at N=2/4; BLAS otherwise | Off |
| `few_orb` | Eigen at N=2/4; BLAS otherwise | On, 8 MiB budget |

The shared helper handles ordinary, batched and prepared convolutions. FewOrb
has no duplicate DLR tables, transformation routines or convolution formula.
Its prepared-operand storage supports edge reuse. Both evaluators already used
persistent scratch and output buffers before this change.

## Main result

Complete order-two topology calls, ranges across self-energy, paired
self-energy and correlator evaluation. A speedup is **baseline time / variant
time**; values below one mean the variant is slower.

| N | Product choice | Rank 13, no reuse | Rank 13, with reuse | Rank 103, no reuse | Rank 103, with reuse |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Eigen | 1.63–1.90× | 2.05–2.28× | 1.42–1.67× | 1.77–2.08× |
| 4 | Eigen | 1.27–1.32× | 1.49–1.57× | 1.16–1.23× | 1.50–1.52× |
| 8 | BLAS | ≈0.98× | 1.08–1.21× | ≈0.99× | 1.15–1.27× |
| 16 | BLAS | 0.98–0.99× | 1.07–1.19× | 0.99–1.00× | 1.13–1.26× |

At N=2/4, selected small products improve the updated baseline, and edge reuse
adds further benefit. At N=8/16 the selected products are BLAS, so the uncached
comparison reveals residual implementation overhead with the same arithmetic.
The earlier large Eigen-related regressions are removed by this crossover.
Uncached N=8/16 remains 0.2–2.1% slower than the baseline; rounding to 1.00×
does not imply exact parity. Cached FewOrb beats the baseline in all 24 complete
order-two cases and improves every median relative to uncached FewOrb.

The incremental cache gains across these cases are N=2: 1.20–1.32×, N=4: 1.13–1.29×, N=8: 1.11–1.28×, N=16: 1.08–1.27×.
These are uncached FewOrb time / cached FewOrb time, isolating reuse with other
choices fixed. Dense versus uncached FewOrb also includes remaining traversal,
validation and workspace scheduling differences; it is not a strict matrix-kernel
A/B. The N=8/16 rows make those differences visible without changing kernels.

The baseline itself has improved. For example, rank-103 N=4 order-two self-energy
changed from 21.362 ms in the [previous sweep](eigen_all_dimensions.md) to
16.898 ms here. That cross-run comparison is indicative, since the trials
were separate. The tables below use variants measured together in this run.
The common DLR improvement is not credited to the small-matrix or reuse columns.

## Complete order-two timings

Median **milliseconds per call**, including backbone preparation and matching
public-call/result-wrapper boundaries. A paired backbone covers both endpoint
directions. Changing N also changes the number of arc flavours, so timings across
dimensions are not pure matrix-size scaling.

### DLR rank 103

| N | Evaluation | Backbones | Baseline | Products, no reuse | Products + reuse | No reuse vs baseline | With reuse vs baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 16 | 3.327 | 2.138 | 1.624 | 1.56× | 2.05× |
| 2 | Paired self-energy | 8 | 1.973 | 1.184 | 0.950 | 1.67× | 2.08× |
| 2 | Correlator | 8 | 1.655 | 1.168 | 0.937 | 1.42× | 1.77× |
| 4 | Self-energy | 32 | 16.898 | 14.278 | 11.110 | 1.18× | 1.52× |
| 4 | Paired self-energy | 16 | 10.848 | 8.816 | 7.251 | 1.23× | 1.50× |
| 4 | Correlator | 16 | 7.846 | 6.756 | 5.239 | 1.16× | 1.50× |
| 8 | Self-energy | 48 | 84.562 | 85.295 | 68.294 | 0.99× | 1.24× |
| 8 | Paired self-energy | 24 | 56.279 | 56.964 | 48.755 | 0.99× | 1.15× |
| 8 | Correlator | 24 | 36.783 | 37.181 | 28.966 | 0.99× | 1.27× |
| 16 | Self-energy | 64 | 542.615 | 544.386 | 449.030 | 1.00× | 1.21× |
| 16 | Paired self-energy | 32 | 385.403 | 386.103 | 339.609 | 1.00× | 1.13× |
| 16 | Correlator | 32 | 218.120 | 219.360 | 172.616 | 0.99× | 1.26× |

### DLR rank 13

| N | Evaluation | Backbones | Baseline | Products, no reuse | Products + reuse | No reuse vs baseline | With reuse vs baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 16 | 0.339 | 0.191 | 0.151 | 1.77× | 2.25× |
| 2 | Paired self-energy | 8 | 0.209 | 0.110 | 0.092 | 1.90× | 2.28× |
| 2 | Correlator | 8 | 0.154 | 0.094 | 0.075 | 1.63× | 2.05× |
| 4 | Self-energy | 32 | 1.594 | 1.240 | 1.034 | 1.29× | 1.54× |
| 4 | Paired self-energy | 16 | 1.094 | 0.832 | 0.734 | 1.32× | 1.49× |
| 4 | Correlator | 16 | 0.656 | 0.517 | 0.419 | 1.27× | 1.57× |
| 8 | Self-energy | 48 | 7.470 | 7.628 | 6.517 | 0.98× | 1.15× |
| 8 | Paired self-energy | 24 | 5.508 | 5.621 | 5.085 | 0.98× | 1.08× |
| 8 | Correlator | 24 | 2.820 | 2.880 | 2.338 | 0.98× | 1.21× |
| 16 | Self-energy | 64 | 49.287 | 49.770 | 43.575 | 0.99× | 1.13× |
| 16 | Paired self-energy | 32 | 38.736 | 39.092 | 36.177 | 0.99× | 1.07× |
| 16 | Correlator | 32 | 16.856 | 17.155 | 14.157 | 0.98× | 1.19× |

## Higher-order results

Complete order-three calls were run at N=2/4. Speedup ranges across the same three
evaluation kinds are:

| N | DLR rank | No reuse vs baseline | With reuse vs baseline |
| ---: | ---: | ---: | ---: |
| 2 | 13 | 1.63–1.79× | 2.23–2.36× |
| 2 | 103 | 1.41–1.56× | 1.91–2.11× |
| 4 | 13 | 1.25–1.27× | 1.56–1.65× |
| 4 | 103 | 1.15–1.19× | 1.55–1.59× |

At N=8/16, order-three and order-five measurements use at most 63 stratified
prepared backbones. The full order-three topology can be requested with
`--full-order3`, but was not run here. Prepared samples exclude backbone
construction. Dense returns one result per backbone; FewOrb returns one span
sum, so these ratios also include the output allocation/accumulation difference.
They do not predict full order-five topology timings.

Rank-103 order-five samples, median **milliseconds per call**:

| N | Evaluation | Backbones | Baseline | Products, no reuse | Products + reuse | No reuse vs baseline | With reuse vs baseline |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Self-energy | 63 | 44.305 | 30.491 | 21.759 | 1.45× | 2.04× |
| 2 | Paired self-energy | 63 | 46.816 | 31.437 | 22.708 | 1.49× | 2.06× |
| 2 | Correlator | 63 | 44.342 | 31.322 | 22.679 | 1.42× | 1.96× |
| 4 | Self-energy | 63 | 104.004 | 90.643 | 65.233 | 1.15× | 1.59× |
| 4 | Paired self-energy | 63 | 113.161 | 97.250 | 71.821 | 1.16× | 1.58× |
| 4 | Correlator | 63 | 102.030 | 89.089 | 63.816 | 1.15× | 1.60× |
| 8 | Self-energy | 63 | 327.814 | 329.919 | 239.596 | 0.99× | 1.37× |
| 8 | Paired self-energy | 63 | 364.882 | 367.317 | 278.132 | 0.99× | 1.31× |
| 8 | Correlator | 63 | 314.127 | 316.587 | 226.446 | 0.99× | 1.39× |
| 16 | Self-energy | 63 | 1445.337 | 1451.328 | 1137.287 | 1.00× | 1.27× |
| 16 | Paired self-energy | 63 | 1670.263 | 1675.866 | 1359.461 | 1.00× | 1.23× |
| 16 | Correlator | 63 | 1338.004 | 1346.692 | 1027.695 | 0.99× | 1.30× |

Rank-13 prepared results and all intermediate order-two/three samples remain in
the [raw CSV](timings_shared_convolution.csv). Reuse savings depend on the number
of repeated edge recipes and the available cache capacity.

## Product crossover

The crossover uses the earlier [isolated production-kernel measurements](eigen_gemm_dimensions.csv)
on the same compiler/CPU setup. Median **nanoseconds per complex product**, with
unit scaling; overwrite computes C=AB and accumulation computes C+=AB:

| N | Eigen overwrite / accumulate | BLAS overwrite / accumulate | Selected |
| ---: | ---: | ---: | --- |
| 2 | 11.6 / 12.1 | 82.7 / 76.3 | Eigen |
| 4 | 63.8 / 67.1 | 129.2 / 116.1 | Eigen |
| 8 | 524.0 / 515.3 | 248.9 / 212.1 | BLAS |
| 16 | 2867.7 / 2833.5 | 983.2 / 862.1 | BLAS |

Eigen and handwritten code use the generic compiler target, whereas OpenBLAS
selects CPU-specific kernels at runtime. Thus N=4 is a measured practical cutoff
for this build, not a universal cutoff. Explicit Eigen and handwritten products
through N=16 remain available for experiments; the evaluator uses BLAS there.
These isolated kernel timings were retained from the previous round, while all
whole-evaluator timings in this report were rerun after sharing the convolution.

## Cache capacity and correctness

Each cached recipe stores coefficients and their Hilbert transform, two complex
`(r,N,N)` tensors. At N=16/rank 103 one recipe uses 843,776 bytes (0.805 MiB), so
nine fit within 8 MiB. The order-five samples reach this capacity:

| Evaluation | Hits | Misses | Hit rate | Entries | Payload (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 411 | 93 | 81.5% | 9 | 7.242 |
| Paired self-energy | 416 | 88 | 82.5% | 9 | 7.242 |
| Correlator | 416 | 88 | 82.5% | 9 | 7.242 |

Once full, the existing entries remain reusable and other recipes are recomputed.
Complete order two needs three entries, so its results do not reflect capacity
pressure. Reported bytes count active numerical payload; keys, fixed scratch,
Action storage and container metadata are additional. Retained allocations can
reflect larger preceding calls but obey the numerical budget.

All **324 evaluator rows / 108 cases** passed finite/nonzero-reference checks and
the relative-error threshold of 2e-10. The maximum relative maximum discrepancy
is **3.31e-14**. The complete rebuilt CPU suite passed **88/88** CTest entries
in 34.68 seconds, including the dense, batched and block-sparse consumers.
Prepared complex rectangular operands are tested on either side, including
cache immutability, callback overwrite/accumulation and packed buffer tails.
Existing tests cover analytic convolutions, in-place solves and output aliasing.

## Reproduction and provenance

Implementation: XCA `64a34f73`. Benchmark driver: `b46b4dd`, unchanged in this
round. New raw measurements: `5aa1cc6a`. Dependencies: TRIQS `2930120`,
NDA `4e59077`, cppdlr `b9b24ab`, Eigen 5.0.1.

AMD Ryzen Threadripper PRO 3945WX, 12 cores / 24 hardware threads, Clang 19.1.1,
Release `-O3 -DNDEBUG -std=c++23`, system OpenBLAS. No native-architecture or
fast-math flags. One thread, pinned to CPU 2, `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, no MPI launcher or GPU. The schedutil governor and boost
remained enabled. No builds, tests or other benchmarks ran during measurement;
the workstation was not isolated from unrelated activity.

Five trials rotate variant order. Calls repeat one to five times, calibrated
toward 5 ms per trial. Numerical cache preparation is included on every call;
only allocations are warmed. Model/Action/evaluator construction and first-use
allocation are outside the timings. Minimum/median/maximum timings, errors and
cache counters are retained in the CSV. These are evaluator timings, not a full
solver iteration or multi-core/MPI scaling measurement.

Build/run instructions and fixture details are in the [generic README](../README.md).
The original [handwritten N=4 report](handwritten_n4.md) and
[Eigen-through-N=16 report](eigen_all_dimensions.md), together with their raw
CSVs, are retained. Further useful comparisons are a native-architecture build,
more diverse pole/topology sets, and a larger cache budget at N=16/rank 103;
these were not added to the present sweep.

# ThinkStation P620 / Threadripper PRO 3945WX

These recorded results retain the API names used when measured: `self_energy`
was the directional algorithm and `pairs` summed both special-arc directions.
The current benchmark uses the paired algorithm under `self_energy`; the tables
and raw results below have not been rerun or relabelled.

Measured on 2026-11-03 with an AMD Ryzen Threadripper PRO 3945WX (12 cores,
24 hardware threads), Clang 19.1.1, a Release `-O3` build, and system OpenBLAS.
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; no GPU or MPI parallelism.
Frequency boost was enabled with the `schedutil` governor, and CPU affinity was
not pinned. The workstation was not isolated from unrelated activity. Builds and
other benchmark runs were kept separate from this measurement.

Evaluator source: XCA `53638c5`, including batched convolution `4e2688f`.
Installed CPU dependencies: TRIQS `2930120`, NDA `4e59077`, cppdlr `b9b24ab`.
The scalar baseline already uses the new fast convolution. These results measure
the additional changes in the batched evaluator; they do not include the older
cppdlr convolution or its former redundant-Hilbert-transform algorithm.

All matrices have Hilbert-space dimension **4**, representing two fermion modes.
The model combines hybridization and two retarded density operators with complex
couplings. The propagator is a deterministic dressed atomic fixture, not a
converged impurity-solver result. Both meshes have beta 2; their DLR ranks are
**13** and **103**. Rank 103 uses dimensionless cutoff 100000 and tolerance 1e-12.

## Prepared numerical evaluations

Times are median **milliseconds per call**, including the sum of all listed
backbones. They include vertex operations, edge kernels, convolution, endpoint
sums, correlator reflection/join where applicable, local accumulation, and
returned-array allocation. Action/evaluator construction, workspace allocation,
and backbone preparation/materialization are outside timing.

Each case uses one fully crossing topology. At order 2 the full index space is
small enough to include every backbone; at orders 3 and 5 the benchmark uses
63 stratified samples with seed 1789. Paired self-energy entries each represent
both distinguished-arc directions, so their backbone counts are not directly
comparable with ordinary self-energy counts at high orders. The listed best
capacity is the fastest median among 1, 2, 4, 8, 16, 32 and 64; it is descriptive,
not proof of a statistically significant optimum.

### DLR rank 13

| Evaluation | Order | Backbones | Scalar fast | Batch 1 | Best capacity | Best time | vs scalar | vs batch 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 2 | 32 | 1.978 | 1.621 | 8 | 1.537 | 1.287× | 1.054× |
| Paired self-energy | 2 | 16 | 1.364 | 1.109 | 8 | 1.066 | 1.280× | 1.041× |
| Correlator | 2 | 16 | 0.900 | 0.672 | 8 | 0.621 | 1.448× | 1.081× |
| Self-energy | 3 | 63 | 5.866 | 5.181 | 64 | 4.822 | 1.216× | 1.074× |
| Paired self-energy | 3 | 63 | 7.712 | 6.369 | 64 | 6.006 | 1.284× | 1.061× |
| Correlator | 3 | 63 | 5.505 | 4.619 | 64 | 4.249 | 1.295× | 1.087× |
| Self-energy | 5 | 63 | 9.995 | 9.156 | 64 | 8.477 | 1.179× | 1.080× |
| Paired self-energy | 5 | 63 | 11.636 | 10.332 | 64 | 9.642 | 1.207× | 1.071× |
| Correlator | 5 | 63 | 9.619 | 8.601 | 64 | 7.891 | 1.219× | 1.090× |

### DLR rank 103

| Evaluation | Order | Backbones | Scalar fast | Batch 1 | Best capacity | Best time | vs scalar | vs batch 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 2 | 32 | 24.205 | 21.488 | 8 | 20.080 | 1.205× | 1.070× |
| Paired self-energy | 2 | 16 | 15.109 | 13.083 | 8 | 12.418 | 1.217× | 1.054× |
| Correlator | 2 | 16 | 11.721 | 10.070 | 8 | 9.217 | 1.272× | 1.093× |
| Self-energy | 3 | 63 | 81.948 | 75.003 | 8 | 69.309 | 1.182× | 1.082× |
| Paired self-energy | 3 | 63 | 95.157 | 84.141 | 32 | 78.676 | 1.209× | 1.069× |
| Correlator | 3 | 63 | 79.558 | 72.327 | 8 | 65.858 | 1.208× | 1.098× |
| Self-energy | 5 | 63 | 146.596 | 139.995 | 64 | 128.625 | 1.140× | 1.088× |
| Paired self-energy | 5 | 63 | 162.211 | 148.393 | 8 | 137.231 | 1.182× | 1.081× |
| Correlator | 5 | 63 | 145.086 | 136.601 | 8 | 124.543 | 1.165× | 1.097× |

## Complete order-2 topology

These calls additionally include preparation of every backbone and the ordinary
block-Green-function result wrapper. Each uses the full 32-contribution
self-energy topology on one process. They remain evaluator timings, not complete
solver iterations.

### DLR rank 13

| Evaluation | Order | Backbones | Scalar fast | Batch 1 | Best capacity | Best time | vs scalar | vs batch 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 2 | 32 | 1.956 | 1.648 | 8 | 1.555 | 1.258× | 1.060× |

### DLR rank 103

| Evaluation | Order | Backbones | Scalar fast | Batch 1 | Best capacity | Best time | vs scalar | vs batch 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 2 | 32 | 23.329 | 21.167 | 8 | 19.652 | 1.187× | 1.077× |

## Interpretation and recommendation

The best batched variant is **1.14–1.45×** as fast as the scalar fast evaluator
across these cases (about 12–31% less elapsed time). However, grouping more than
one backbone improves on the batched implementation at capacity 1 by only
**1.04–1.10×** (about 4–9% less time). The larger combined gain must not be
attributed entirely to batching the DLR transforms.

Capacity 1 uses the batched evaluator's other changes: direct GEMM into reusable
vertex/endpoint storage and one returned result for a collection of prepared
backbones. The scalar prepared baseline calls and sums individual owning results.
For larger capacities, DLR solves and transforms share their fixed matrices across
batch entries, and correlator reflection also runs across the batch. The small
matrix products and endpoint sums still execute separately for each backbone.
Thus the comparison demonstrates some benefit from wider shared transforms,
while substantial work remains unchanged. It does not isolate convolution alone.

**Use batch capacity 8 as the first setting to try**, retaining the scalar
implementation as the baseline. Capacity 8 is within **0.6%** of the best median
in every measured case, including both complete-topology cases. Larger capacities
provide no convincing additional gain here and increase retained batch storage.
At rank 103 and dimension 4, the eight batch-sized complex buffers (three running
buffers plus five convolution scratch buffers) account for approximately
**1.61 MiB at capacity 8**, versus **12.88 MiB at capacity 64**. These are calculated
buffer sizes, not measured peak memory; Action data, endpoint storage, cached DLR
matrices and temporary reflection/results are additional.

The complete order-2 topology gains are **1.26× at rank 13** and **1.19× at rank
103** versus scalar fast, showing that preparation and result wrapping do not erase
the benefit in this small case. Higher-order complete-topology timings are not
inferred from the bounded samples.

The batching API is currently C++ only, with no Python wrapper. All sampled
backbones in a call share order and distinguished-vertex position, allowing them
to form a batching cohort. Mixed topologies or orders may shorten cohorts; actual
MPI work partitions may also yield smaller tails. The results support keeping the
implementation as a measurable CPU experiment and testing representative solver
workloads before selecting it as a default.

## Validation and measurement uncertainty

All **20 cases × 8 variants = 160 rows** passed finite, nonzero and relative
maximum-error checks before and after timed repeated calls. The largest relative
maximum discrepancy from scalar fast was **4.18e-16**, below the `2e-10` threshold.
This validates the measured workloads; independent convolution and evaluator
unit tests cover additional shapes, direction/statistics combinations and reuse.

Every median uses five trials with rotating variant order. Rank-13 trials use
one to three calls according to a short calibration; all rank-103 trials use one
call. Minimum/maximum times and repeat counts are retained in
[timings.csv](timings.csv). Across all rows, median trial spread `max/min - 1`
was **0.46%**, its 90th percentile was approximately **1.41%**, and the largest
spread was **15.12%**. This was one full sweep. Differences of a few tenths of a
percent should not be treated as portable or significant capacity rankings.

Useful follow-up measurements are a complete higher-order topology of a bounded
model, the actual rank/order/operator mixture used by the solver, and the effect
of mixed-cohort input and smaller MPI-local work sets. A standalone convolution
profile would separate transform savings from vertex and endpoint costs. No
claim is made about full-solver speedup, other CPUs, or GPU performance.

See the [generic benchmark README](../README.md) for model construction, execution
instructions, sampling, and the exact timing boundaries.

The [small-N follow-up analysis](small_n_analysis.md) profiles the limited
batching gain and measures additional CPU arithmetic and reuse experiments.

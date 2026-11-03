# ThinkStation P620 / Threadripper PRO 3945WX

Single-threaded CPU measurements on 2026-11-03, using Clang 19.1.1 Release
(`-O3`), system OpenBLAS, and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Dependencies: TRIQS `2930120`, NDA `4e59077`, fixed cppdlr `b9b24ab`.
CPU affinity was not pinned; boost and the `schedutil` governor were enabled.
Builds and other benchmark runs were kept separate from these measurements.
The workstation was not isolated from unrelated activity.

These are complete **prepared-backbone evaluator calls**, including discovery,
numerical work, joins, owning output and existing reset behavior. They exclude
backbone/model construction and are neither complete topology sums nor solver
iterations. See the [generic README](../README.md) for the exact workloads,
bounded selection procedure, validation and commands.

| Variant | Source checkpoint | Storage and arithmetic | Results |
| --- | --- | --- | --- |
| Baseline | `75a7c58` (sparse implementation `78576a6`) | Padded views, allocating vertex products, fixed cppdlr | [baseline.csv](baseline.csv) |
| Output GEMM | `6565624` | Padded views, reused vertex/endpoint outputs, active-only self-energy scaling, fixed cppdlr | [output_gemm.csv](output_gemm.csv) |
| Packed, fixed convolution | `1564a0e` | Contiguous active views over reusable capacity buffers, fixed cppdlr | [packed_fixed.csv](packed_fixed.csv) |
| Packed, fast convolution | `fb4b4aa` | Packed buffers, output-parameter convolution, immutable convolution tables in Action and mutable workspace in the evaluator | [packed_fast.csv](packed_fast.csv) |

The baseline was compiled against frozen headers and static libraries from the
validated structural checkpoint. The same driver, model data and selected
backbones were used for comparison. All **56** cases passed elementwise checks
before and after repeated evaluation. Both output GEMM and packed/fixed matched
the baseline exactly on these samples. The maximum relative error with the final
fast convolution was **2.618e-13**, measured against each backbone's maximum
reference magnitude; zero outputs stayed exactly zero. The final production
checkpoint also passed all **86 CTest entries**, including Python and real
three-rank MPI tests.

## Productive calls

Times below are median **microseconds per backbone**, baseline → packed/fast.
Speedup is baseline time divided by final time. Each row measures eight selected
productive backbones. The model names and selection procedure are described in
the [generic README](../README.md).

### DLR rank 13

| Model | Order | Self-energy, µs | Speedup | Correlator, µs | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `density4` | 2 | 267.90 → 183.39 | 1.461x | 223.77 → 169.91 | 1.317x |
| `density4` | 3 | 168.74 → 114.93 | 1.468x | 206.77 → 146.82 | 1.408x |
| `density4` | 5 | 161.80 → 110.85 | 1.460x | 213.50 → 147.80 | 1.445x |
| `unequal_sets_nA0` | 2 | 156.88 → 102.24 | 1.534x | 124.59 → 93.34 | 1.335x |
| `unequal_sets_nA0` | 3 | 174.12 → 111.59 | 1.560x | 136.83 → 93.47 | 1.464x |
| `unequal_sets_nA0` | 5 | 145.90 → 93.26 | 1.564x | 187.85 → 125.70 | 1.494x |
| `n_partition_spin_xyz` | 2 | 437.22 → 285.00 | 1.534x | 349.56 → 272.72 | 1.282x |
| `n_partition_spin_xyz` | 3 | 652.43 → 418.36 | 1.559x | 448.91 → 329.40 | 1.363x |
| `n_partition_spin_xyz` | 5 | 838.34 → 529.68 | 1.583x | 664.50 → 458.15 | 1.450x |

Across these 18 cases, the speedup ranges from **1.282x to 1.583x**, with an
unweighted median of **1.462x**.

### DLR rank 103

| Model | Order | Self-energy, µs | Speedup | Correlator, µs | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `density4` | 2 | 3487.73 → 2246.44 | 1.553x | 2673.66 → 1930.60 | 1.385x |
| `density4` | 3 | 2215.13 → 1453.79 | 1.524x | 2628.36 → 1795.51 | 1.464x |
| `density4` | 5 | 2171.95 → 1423.65 | 1.526x | 2790.45 → 1865.57 | 1.496x |
| `unequal_sets_nA0` | 2 | 1949.35 → 1246.18 | 1.564x | 1443.82 → 1045.64 | 1.381x |
| `unequal_sets_nA0` | 3 | 2228.61 → 1424.90 | 1.564x | 1719.58 → 1153.69 | 1.491x |
| `unequal_sets_nA0` | 5 | 1869.04 → 1196.01 | 1.563x | 2408.92 → 1577.70 | 1.527x |
| `n_partition_spin_xyz` | 2 | 5511.05 → 4086.51 | 1.349x | 4099.93 → 3323.66 | 1.234x |
| `n_partition_spin_xyz` | 3 | 8895.63 → 6640.97 | 1.340x | 5799.23 → 4557.42 | 1.272x |
| `n_partition_spin_xyz` | 5 | 11849.89 → 8843.99 | 1.340x | 9166.28 → 7057.69 | 1.299x |

Across these 18 cases, the speedup ranges from **1.234x to 1.564x**, with an
unweighted median of **1.477x**. The maximum-dimension-six model gains less at
rank 103 than the dimension-one/two models. This is a measured whole-call
difference; these timings alone do not identify its remaining bottleneck.

## What each change contributes

Each entry is the median of the 18 productive case ratios at that rank. Columns
compare successive implementations, while the last compares baseline to final;
the medians therefore need not multiply to the final median.

| Rank | Baseline → output GEMM | Output GEMM → packed fixed | Packed fixed → packed fast | Baseline → final |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 1.043x | 1.018x | 1.369x | 1.462x |
| 103 | 1.033x | 1.003x | 1.431x | 1.477x |

Every value is an earlier-time/later-time **speedup**. Reusing vertex/endpoint
outputs and limiting self-energy scaling to the active shape gives a modest improvement.
Packing alone adds little at rank 103, but supplies the contiguous operands
needed by the fast output-parameter convolution. The largest measured increment
is replacing fixed cppdlr convolution with that routine and its reusable
workspace. These controls isolate cumulative implementation stages, not
independent benefits that can be added together. The complete evaluator still
allocates owning results and performs other orchestration work.

## Calls with no completed contribution

These 20 cases are separate from the productive-call summaries. Each table
entry again gives baseline → final microseconds per backbone, using eight
selected backbones. There were no selected zero-completion cases at order two
or for `n_partition_spin_xyz` at order three.

Self-energy cases have no completed paths and do no convolution:

| Model | Order | Rank 13, µs | Rank 103, µs |
| --- | ---: | ---: | ---: |
| `density4` | 3 | 2.316 → 2.565 | 8.298 → 8.769 |
| `density4` | 5 | 2.447 → 2.418 | 8.345 → 8.341 |
| `unequal_sets_nA0` | 3 | 1.222 → 1.212 | 5.494 → 5.527 |
| `unequal_sets_nA0` | 5 | 1.361 → 1.419 | 5.650 → 5.535 |
| `n_partition_spin_xyz` | 5 | 4.197 → 5.019 | 27.738 → 28.314 |

Their median speedups are **0.959x at rank 13** and **0.994x at rank 103**:
three of five cases regress at each rank. The largest absolute regression is
**0.822 µs (19.6%)**, for the rank-13, order-five number-sector model. At rank
103 the largest relative regression is 5.7% (`density4`, order three), and the
largest absolute regression is 0.576 µs (number sectors, order five). These are
small absolute costs, but they matter if a workload contains mostly rejected
self-energy backbones; productive-call speedups cannot be applied to them.

Zero-completion correlators still evaluate segment paths before finding that
no join contributes. The segment counts below are totals over the eight
backbones, and are identical at the two ranks:

| Model | Order | Segments | Rank 13, µs | Rank 103, µs |
| --- | ---: | ---: | ---: | ---: |
| `density4` | 3 | 48 | 127.919 → 85.785 | 1653.745 → 1100.637 |
| `density4` | 5 | 12 | 61.709 → 42.986 | 816.672 → 544.545 |
| `unequal_sets_nA0` | 3 | 12 | 35.092 → 23.691 | 459.697 → 298.966 |
| `unequal_sets_nA0` | 5 | 8 | 47.344 → 31.355 | 611.750 → 396.010 |
| `n_partition_spin_xyz` | 5 | 32 | 415.816 → 268.714 | 6120.296 → 4589.846 |

All five improve at each rank, with median speedups **1.491x** and **1.503x**,
respectively. A zero output therefore does not imply a cheap call in the current
correlator implementation.

## Storage cost and interpretation

The fast scalar convolution retains five complex workspace buffers in each
evaluator. With rank `r` and maximum block dimension `Nmax`, their capacity cost
is `5 * r * Nmax² * 16` bytes, in addition to the evaluator's other buffers.
The immutable convolution tables cost `3 * r² * 16` bytes per Action. At rank
103 this is:

| Maximum block dimension | Extra evaluator workspace | Action convolution tables |
| ---: | ---: | ---: |
| 1 | 8,240 bytes | 509,232 bytes |
| 2 | 32,960 bytes | 509,232 bytes |
| 6 | 296,640 bytes | 509,232 bytes |

At maximum dimension 64, the same rank requires about **32.2 MiB** of retained
convolution workspace. Larger blocks therefore need explicit memory budgeting;
the timings above only cover maximum dimensions one, two and six. Workspace
reuse replaces transient convolution allocations with persistent capacity, so
this table is not a measurement of the change in peak process memory.

[selected_backbones.csv](selected_backbones.csv) records each ordinal, discovered
path/segment counts and output magnitude. The sample intentionally separates
productive calls from zero-completion calls; it does not estimate their frequency
in a solver run. These are not weighted workload summaries, complete topology
sums, or full solver speedups. Three timing trials per row give
median/minimum/maximum values in the CSVs; there are no confidence intervals.
Small differences, especially the packing-only and zero-path changes, need
repeat measurement before being treated as portable results. The rank-13
number-sector zero-path regression has nonoverlapping trial ranges in this run,
so it should not simply be dismissed as timer noise.

The local elementwise reference and statically linked baseline executable are
also saved in `cbuild/benchmarks/block_sparse_convolution/reference/` (ignored
build artifacts). Rebuilding the documented source checkpoint and rerunning
record mode regenerates a reference if these artifacts are cleaned.

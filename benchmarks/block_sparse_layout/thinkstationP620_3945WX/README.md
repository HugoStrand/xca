# ThinkStation P620 / Threadripper PRO 3945WX

Measured on 2026-11-03 using an AMD Ryzen Threadripper PRO 3945WX (12 cores,
24 hardware threads), Clang 19.1.1, Release `-O3`, and system OpenBLAS.
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; no GPU or MPI parallelism.
Frequency boost was enabled with the `schedutil` governor; CPU affinity was not
pinned. Other workstation activity was not isolated.

XCA base: `dev_backbone`, `c650fe0`, plus the uncommitted experiment and
fast-convolution draft. Installed CPU dependencies: TRIQS `2930120`, NDA
`4e59077`, cppdlr `b9b24ab`. The cppdlr reference already contains the fix for
the redundant Hilbert transforms. These comparisons do not include the much
slower pre-fix cppdlr algorithm.

The DLR basis has **rank 103**, inverse temperature two, cutoff `100000`, and
tolerance `1e-12`. Each value below is the median **microseconds per prepared
numerical segment**, including initial active-data copy, vertex products,
applicable vertex kernels, edge copies, convolutions and forward scaling.
Convolution includes both values-to-coefficients transforms and output
assignment. Setup and workspace allocation are outside timing.

These are not complete diagram evaluator timings. Planning, operator lookup,
edge-kernel multiplication, endpoint sums, correlator reflection/join, reset
and final accumulation are excluded. Physical fixtures use the test models'
actual block paths and operators, with deterministic dressed atomic
propagators from the test helper, rather than converged solver outputs.

## Prepared paths from physical test models

| Fixture | Traversed dimensions | Capacity dimension | Current | Reused GEMM, padded | Reused GEMM, packed | Packed + fast | Combined speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Density, forward | 1:1:1 | 1 | 265.30 | 256.34 | 256.92 | 171.12 | 1.55× |
| Density, backward | 1:1:1 | 1 | 262.53 | 256.33 | 256.46 | 170.37 | 1.54× |
| Unequal sets, forward | 2:1:2 | 2 | 306.51 | 296.79 | 296.87 | 194.33 | 1.58× |
| Unequal sets, backward | 1:2:1 | 2 | 281.74 | 273.10 | 273.18 | 178.44 | 1.58× |
| Number symmetry, forward | 6:4:6 | 6 | 1019.22 | 1005.00 | 1003.68 | 795.44 | 1.28× |
| Number symmetry, backward | 4:6:4 | 6 | 863.59 | 843.93 | 843.29 | 657.91 | 1.31× |

`density4` has only one-dimensional sectors. `unequal_sets_nA0` reaches
dimension two. `n_partition_spin_xyz` has sectors of dimensions `1,4,6,4,1`.
Both sides of an order-three correlator are measured separately, each with
two vertex/convolution steps. Selection deliberately favors dimension changes
and larger blocks; these are not averages weighted by occurrence in a solver.

Replacing allocating `matmul` with reusable GEMM output improves these full
segments by only **1.01–1.03×**. Packing after that changes their timings by
less than 0.3%, too little to support a separate performance claim. Combined
with fast convolution, the improvement is **1.28–1.58×** (22–37% less time).

## Synthetic shape stress cases

| Fixture | Traversed dimensions | Capacity dimension | Current | Reused GEMM, padded | Reused GEMM, packed | Packed + fast | Combined speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform 1, forward | 1:1:1 | 1 | 265.17 | 256.86 | 256.86 | 170.86 | 1.55× |
| Uniform 4, forward | 4:4:4 | 4 | 694.71 | 681.01 | 680.90 | 512.57 | 1.36× |
| Anchor 1, forward | 1:4:16:4:1 | 64 | 2178.70 | 2143.45 | 2099.78 | 1738.66 | 1.25× |
| Anchor 4, forward | 4:1:16:4:1 | 64 | 2666.22 | 2612.49 | 2561.87 | 2119.62 | 1.26× |
| Reaches 64, forward | 4:16:64:16:4 | 64 | 39831.87 | 39388.70 | 38954.53 | 25465.03 | 1.56× |
| Whole-buffer scaling control | 1:4:16:4:1 | 64 | 2569.16 | 2156.90 | 2107.81 | 1746.94 | 1.47× |
| Uniform 1, backward | 1:1:1 | 1 | 262.81 | 256.80 | 257.20 | 171.23 | 1.53× |
| Uniform 4, backward | 4:4:4 | 4 | 695.25 | 684.03 | 679.80 | 513.23 | 1.35× |
| Anchor 1, backward | 1:4:16:4:1 | 64 | 2091.74 | 2067.12 | 2034.06 | 1671.05 | 1.25× |
| Anchor 4, backward | 4:1:16:4:1 | 64 | 2662.35 | 2620.95 | 2573.44 | 2136.09 | 1.25× |
| Reaches 64, backward | 4:16:64:16:4 | 64 | 39349.19 | 39234.44 | 39080.27 | 25468.96 | 1.54× |

Uniform cases have two steps; mixed cases have four. The dimension sequence
lists the initial sector and the sector after each vertex. For forward
traversal the initial dimension fixes the running product's column count;
for backward traversal it fixes the row count. A capacity of 64 with a
maximum visited dimension of 16 models a larger sector that this selected
path does not visit. These cases are controlled numerical inputs, not physical
symmetry decompositions or projections for a ten- or fourteen-mode model.

On the mixed small paths, packing after reusable GEMM saves about 1.6–2.0%
of segment time. Combined improvement with fast convolution is **1.25–1.26×**.
The paths reaching dimension 64 improve by **1.54–1.56×**. Fast convolution
accounts for most of the gain.

The whole-buffer scaling control isolates a separate production self-energy
behavior. Its `padded_matmul_active` result is **2192.77 us**, versus
**2569.16 us** for scaling the entire padded buffer: **1.17×** speedup from
active-only scaling, without changing the layout or multiplication method.
The physical correlator fixtures already scale only active data and do not
have this overhead. The full control's **1.47×** combined speedup must not be
attributed to packing alone.

## Vertex-only diagnostic

This phase omits every edge copy and convolution, retaining initial copy,
vertex operations and forward scaling. It reveals savings that the dominant
convolution hides in the full numerical segments. These numbers are not
additional speedups to multiply into the tables above.

| Fixture | Traversed dimensions | Capacity dimension | Current | Reused GEMM, padded | Reused GEMM, packed | Packed + fast | Combined speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Density, forward | 1:1:1 | 1 | 24.09 | 15.72 | 15.72 | 15.71 | 1.53× |
| Density, backward | 1:1:1 | 1 | 21.63 | 15.63 | 15.51 | 15.57 | 1.39× |
| Unequal sets, forward | 2:1:2 | 2 | 26.58 | 16.97 | 16.90 | 16.96 | 1.57× |
| Unequal sets, backward | 1:2:1 | 2 | 24.22 | 15.74 | 15.80 | 15.75 | 1.54× |
| Number symmetry, forward | 6:4:6 | 6 | 51.05 | 34.24 | 34.17 | 34.06 | 1.50× |
| Number symmetry, backward | 4:6:4 | 6 | 48.72 | 31.31 | 31.14 | 31.23 | 1.56× |

Reusable GEMM output improves the physical vertex-only paths by
**1.38–1.57×**. Packing has little further effect at these dimensions. In the
synthetic mixed-small cases it saves roughly another 8–13% of time in that
phase, while convolution limits its effect on the complete numerical segment.

## Validation and uncertainty

All **17 fixtures × two phases × five variants** passed elementwise finite
checks and relative maximum-error checks before and after repeated calls.
The largest relative maximum error was **4.67e-14** overall and **8.04e-15**
on physical fixtures, below the `2e-10` threshold. The output-GEMM and packed
fixed-cppdlr variants had zero measured discrepancy from the baseline.
A separate preflight checks the maximum-capacity convolution against the
original fast draft across changing rectangular shapes and legal output/input
aliasing. This validates the numerical replay, not unimplemented endpoint or
correlator-join changes in a production evaluator.

Each reported median uses five trials, with adaptive repetitions targeting
8 ms per trial and rotating variant order. The raw CSV records repetitions,
minimum and maximum times. Across all rows the median trial spread
`max/min - 1` was 0.75%; its 90th percentile was 3.75%, and the largest outlier
spread was 17.1%. Small percentages should not be treated as precise portable
speedups. An earlier full sweep reproduced the same overall ranking; it is
retained in [timings_initial.csv](timings_initial.csv). The final results with
stricter elementwise-finite and relative-error validation are in
[timings.csv](timings.csv).

## Recommended implementation

Adopt **packed active views into reusable capacity buffers, output-parameter
GEMMs, and fast convolution together**, in reviewable stages:

1. Replace vertex `matmul` temporaries with output GEMM into the existing
   spare buffer, swap storage roles, and restrict operations to active data.
   Keep current convolution while validating the numerical behavior.
2. Change active views to contiguous `(r,rows,columns)` views of the same
   reserved capacity. Validate growing and shrinking dimensions, both
   traversal directions, saved endpoint data, and preservation of the forward
   result while computing the backward correlator segment. This step provides
   a simple layout for the fast convolution API; its standalone timing benefit
   is small.
3. Put immutable fast-convolution tables in `Action` and reusable scratch in
   each evaluator. Size scratch capacity from the maximum block dimension;
   dense evaluation uses the same interface with a fixed dimension. Avoid a
   cache or allocation per shape. Replace the convolution calls and measure
   full self-energy and correlator evaluations using the existing test models.

The existing spare edge buffer can hold vertex output, so packed dimension
changes do not themselves require another full-size running-result buffer.
Fast convolution has five reusable scratch buffers: at rank 103 and capacity
64 these total **32.2 MiB**, plus small DLR tables and the evaluator's other
storage. This is retained storage, not a measurement of total peak memory or
an assertion that it is five buffers more than cppdlr's transient peak.
Capacity allocation may be excessive for very large blocks when most paths
are tiny; evaluate that memory tradeoff separately before applying the design
to the largest physical models.

The next integration check should time complete self-energy and correlator
calls at several orders, and include endpoint sums, reflection/join, and reset.
That will establish how much of the measured segment improvement survives
outside the numerical kernels. The current evidence does not establish an
end-to-end solver speedup or performance on other CPUs.

See the [generic benchmark README](../README.md) for exact variant definitions,
fixture construction, build commands and the proposed ownership boundary.

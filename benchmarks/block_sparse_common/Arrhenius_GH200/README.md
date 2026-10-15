# Arrhenius / GH200 diagram evaluator timings

CPU and CUDA measurements below run on Arrhenius GH200 nodes: NVIDIA GH200 with
an Arm Grace CPU, GCC 14.3 and CUDA 12.9. CPU controls use one OpenMP/BLAS thread.
Times are median synchronous evaluator-call latencies in milliseconds, including
host/device transfers and excluding constructor/model preparation and warm-up.
The DLR rank is 23, with beta = 2, omega_max = 20 and epsilon = 1e-10.

## Production routes and CPU controls

Campaign [finalists-01](finalists-01/README.md), source `d9fa8a2` / runtime `85ecb2d`,
compares six workloads and seven backends. Each case/backend has three fresh
processes with three timed calls per process. The table uses the median of the
three process medians. Individual samples, process ranges, setup times and errors
are in [samples.csv](finalists-01/samples.csv),
[summary.csv](finalists-01/summary.csv) and
[manifest.json](finalists-01/manifest.json).

Each row evaluates one index of one crossing topology: index 0 except the
four-mode occupation OCA row, which uses index 1. NCA and OCA mean expansion orders
1 and 2. `se` denotes self-energy; `spgf` below denotes the single-particle Green's
function. A CPU/CUDA ratio above 1 means CUDA is faster.

| Modes / partition / observable / order | Original CPU ms | CPU control ms | CUDA route | CUDA ms | CPU control / CUDA |
| --- | ---: | ---: | --- | ---: | ---: |
| 4 / occupation / se / NCA | 0.212 | 0.171 | Real NCA | 0.200 | 0.854 |
| 8 / occupation / se / NCA | 29.699 | 3.533 | Real NCA | 1.352 | 2.613 |
| 4 / occupation / se / OCA | 1.206 | 0.475 | Ordered | 14.192 | 0.033 |
| 6 / number / se / OCA | 17.797 | 5.617 | Mapped | 10.660 | 0.527 |
| 8 / number / se / NCA | 92.510 | 92.230 | Mapped | 13.268 | 6.951 |
| 6 / mixed / retarded correlator / O2 | 7.915 | 4.328 | Ordered | 25.843 | 0.167 |

The CPU control is scalar coefficient preparation (`cpu_constants`) for the two
occupation NCA rows, and transform-hoisted convolution (`cpu_hoisted`) for the
other rows. Keeping both CPU columns distinguishes CPU algorithm improvements
from the CUDA comparison.

CUDA routes are explicit production choices: **Real NCA** is the guarded real
scalar batch; **Mapped** combines ordered sparse operators with eligible mapped
special-vertex mixing; **Ordered** uses ordered sparse operators and reference
special mixing. The table uses Real NCA for eligible scalar NCA, Mapped for the
number-sector self-energies, and Ordered for the singleton-set/higher-order and
correlator cases. This is not an automatic dispatch policy. The full
[seven-backend table](finalists-01/README.md) also includes the default library
route and each policy's fallback timings.

Real NCA is about 2.61 times faster than its CPU control for eight-mode occupation
NCA. Mapped number-sector NCA is about 6.95 times faster than the hoisted CPU
control. CPU controls remain faster in the other four selected rows.

## Broader reference calls, including N10

Campaign [initial-01](initial-01/README.md), source `3f462c3`, compares the unchanged
CPU evaluator with library and ordered CUDA. Each case has one process per backend
and three measured repetitions. These are separate measurements from the production
cohort above. The table retains both CUDA routes and ratios against the original
CPU; [summary.csv](initial-01/summary.csv),
[samples.csv](initial-01/samples.csv) and
[manifest.json](initial-01/manifest.json) contain the underlying records.

| Modes / partition / observable / order | CPU ms | Library CUDA ms | Ordered CUDA ms | CPU / ordered CUDA |
| --- | ---: | ---: | ---: | ---: |
| 4 / occupation / se / O3, index 0 | 0.049 | 0.538 | 0.548 | 0.089 |
| 4 / occupation / se / O3, index 1 | 0.815 | 15.099 | 8.511 | 0.096 |
| 4 / mixed / retarded correlator / O2 | 1.059 | 18.155 | 11.039 | 0.096 |
| 6 / number / retarded spgf / O2 | 22.754 | 141.426 | 67.671 | 0.336 |
| 6 / number / se / O3 | 27.965 | 68.535 | 26.392 | 1.060 |
| 8 / number / se / NCA | 98.369 | 192.372 | 67.557 | 1.456 |
| 10 / number / se / NCA | 4827.384 | 415.902 | 169.533 | 28.475 |

All rows use a single index and crossing topology; the two scalar rows state their
indices explicitly and the others use index 0. The scalar index-0 window has zero
signal and no accepted paths, so it measures pruning/orchestration. N10 number
sectors reach maximum block dimension 252: ordered CUDA takes about 170 ms versus
4.83 s for the original CPU. This cohort does not include a threaded or hoisted
CPU control for that row.

## Detailed reports and raw records

- [MPI ownership and phase timings](mpi-01/README.md): per-rank evaluation,
  packing, waiting and reductions for fixed topology/index windows.
- [Real scalar NCA](real-nca-01/README.md): CPU constants, general ordered,
  prototype and production real-NCA calls.
- [Scalar batching](scalar-01/README.md),
  [matched density-set batching](density-matched-01/README.md), and
  [resident density weights](density-resident-01/README.md): separate experimental
  comparisons; these candidates are distinct from the production routes above.
- Raw backend CSVs live under
  [block_sparse_cpu/Arrhenius_GH200](../../block_sparse_cpu/Arrhenius_GH200) and
  [block_sparse_cuda/Arrhenius_GH200](../../block_sparse_cuda/Arrhenius_GH200), using
  the same campaign names. Shared methodology is in [the common README](../README.md).

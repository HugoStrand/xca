# ThinkStation P620 / RTX A4000 measurements

Lenovo ThinkStation P620 (ts), AMD Threadripper PRO 3945WX (12 cores),
NVIDIA RTX A4000 (16 GB, compute capability 8.6), CUDA 13.4 and driver
580.178.04. Measurements use clang-19 Release, TRIQS 4.0, and one
OpenMP/OpenBLAS thread unless the run manifest states otherwise.
GH200-class GPUs are the optimization target; these results are local qualification.

`baseline/` contains seven matched physical-fixture datasets: three quick process
repeats, throughput, method/order coverage, rank-26 and CPU-prefix controls. Raw
samples retain both backends where applicable; select `backend=cpu` for host timing.
Each run records commands, exact indices, source/binary hashes and libraries.

The small N4/rank16 order-eight one-index CPU-only-prefix control measured 1.016 ms,
compared with about 1.033 ms in the CUDA-enabled executable. In the larger N32
workload, `cpu_prefix_check.csv` measured about twice the host evaluator time of the
matched CUDA-enabled run; do not combine incompatible prefix controls to inflate
speedups. Use matched environments and inspect manifests.

`cpu_baseline.csv` uses beta=2, cutoff=40, rank=26, two poles, dimensions 2/4/16/32/64,
and up to eight indices at orders one through three. It records construction and
first/warm primitive/API measurements. Its convolution-hoisting control measured
7.12 / 1.77 ms at N32 and 30.73 / 10.56 ms at N64 for current / hoisted transforms.
The reconstruction differences were zero in these runs. These are host algorithm
controls, not GPU offload speedups; finite samples do not establish confidence intervals.

[Matched comparison reports](../../dense_common/thinkstationP620_RTXA4000/README.md).

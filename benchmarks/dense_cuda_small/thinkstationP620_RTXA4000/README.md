# ThinkStation P620 / RTX A4000 measurements

Lenovo ThinkStation P620 (ts), AMD Threadripper PRO 3945WX (12 cores),
NVIDIA RTX A4000 (16 GB, compute capability 8.6), CUDA 13.4 and driver
580.178.04. Measurements use clang-19 Release, TRIQS 4.0, and one
OpenMP/OpenBLAS thread unless the run manifest states otherwise.
GH200-class GPUs are the optimization target; these results are local qualification.

`results/` contains native-small samples, summaries/manifests, runtime/profiling
records, larger-dimension library controls and validation logs. Run manifests identify
the source version, exact indices, numerical comparisons and environment for each dataset.

Rank16, capacity128, ordinary/paired order-six/eight controls cover dimensions four
and eight, with 128/129 indices and three fresh processes. The 128-index native timings
are approximately 6.9/9.2 ms at N4 and 20.2/27.7 ms at N8 for ordinary order-six/eight
calls, with about 13–15x matched CPU speedups. These results do not imply a crossover
for every method, dimension, rank or dynamic-channel count.

The `native-profile-after-reuse/` ordinary N4/order-six/128-index trace records no
CUDA allocation/free APIs in its warm call. It includes one 4096-byte H2D upload,
one 4096-byte D2H output copy and one 4096-byte unified-memory status migration in
each direction. The reference `native-profile-before-reuse/` capture records the
same explicit transfer sizes plus two 4096-byte allocations/frees. The two captures
use different source revisions identified by their manifests. This residency evidence
is limited to the profiled case; it does not cover every cache transition.

Larger dimensions 16/32/64 use native sequential library operations in the recorded
controls. Their successful comparisons do not qualify larger fixed-size kernels.
[Native timings](../../dense_common/thinkstationP620_RTXA4000/NATIVE_RESULTS.md),
[larger-dimension controls](../../dense_common/thinkstationP620_RTXA4000/LARGE_RESULTS.md),
and [Solver measurements](../../dense_common/thinkstationP620_RTXA4000/SOLVER_RESULTS.md)
provide workload details and numerical scope.

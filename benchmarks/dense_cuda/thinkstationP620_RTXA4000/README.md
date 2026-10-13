# ThinkStation P620 / RTX A4000 measurements

Lenovo ThinkStation P620 (ts), AMD Threadripper PRO 3945WX (12 cores),
NVIDIA RTX A4000 (16 GB, compute capability 8.6), CUDA 13.4 and driver
580.178.04. Measurements use clang-19 Release, TRIQS 4.0, and one
OpenMP/OpenBLAS thread unless the run manifest states otherwise.
GH200-class GPUs are the optimization target; these results are local qualification.

Profiling used Nsight Systems 2026.3.2.476 with CUDA runtime/driver activity,
memory usage and unified-memory faults; CPU sampling/context switches were disabled.
The [portable profiling instructions](../README.md) describe the current commands.

| Trace | Traversals | Visible allocated peak (bytes) | Max UVM migration (bytes) |
|---|---:|---:|---:|
| [N32/r26](trace_n32_r26.csv) | 18 | 39,021,389 | 4,096 |
| [Dynamic/reuse](trace_dynint.csv) | 12 | 29,816,293 | 4,096 |
| [N128/r26](trace_n128_r26.csv) | 18 | 194,503,565 | 61,440 |

All captured inner traversals have zero allocation/deallocation events and zero
explicit host/device copies, including first/warm ordinary, paired and single-particle
calls at N32 and N128, and mixed-dynamic correlator/reuse cases. Unified-memory
migrations remain visible; large work-tensor copies are device-to-device. Visible
peaks include observable library/static allocations and do not prove physical UVM
residency or unreported context/cache memory.

[Phase detail](trace_phases_detail.csv) and [traversal audit](trace_phases_traversal.csv)
cover N32/r26: GEMM-family kernels account for about 65% of active kernel time,
solve-family kernels about 29%. These are kernel-time shares, not whole-call fractions;
launch and synchronization overhead must be judged from wall times. GEMM classification
includes split-K reductions; solve classification includes TRSM/pivot/status kernels.
Other kernels include GEAM, DGMM and vector operations.

[Matched CPU/cuBLAS measurements](../../dense_common/thinkstationP620_RTXA4000/dense_cuda/results.md).

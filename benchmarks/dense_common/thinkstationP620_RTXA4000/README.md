# ThinkStation P620 / RTX A4000 measurements

Lenovo ThinkStation P620 (ts), AMD Threadripper PRO 3945WX (12 cores),
NVIDIA RTX A4000 (16 GB, compute capability 8.6), CUDA 13.4 and driver
580.178.04. Measurements use clang-19 Release, TRIQS 4.0, and one
OpenMP/OpenBLAS thread unless the run manifest states otherwise.
GH200-class GPUs are the optimization target; these results are local qualification.

The reports compare matching workload/backend records; individual manifests identify
source versions and measurement conditions. Medians and ranges describe finite samples,
not confidence intervals. Hardware or prefix changes require fresh matching controls.

- [N4 CPU/cuBLAS reference](RESULTS.md)
- [Native-small workload repeats](NATIVE_RESULTS.md)
- [Larger-dimension library controls](LARGE_RESULTS.md)
- [Complete Solver workload](SOLVER_RESULTS.md)
- [Large-matrix cuBLAS measurements](dense_cuda/results.md)

Raw [CPU/reference datasets](../../dense_cpu/thinkstationP620_RTXA4000/baseline/)
and [native/library datasets](../../dense_cuda_small/thinkstationP620_RTXA4000/results/)
stay with their workloads. `solver-workload-current/` contains the complete Solver
control, and the other CSVs here are derived comparisons.

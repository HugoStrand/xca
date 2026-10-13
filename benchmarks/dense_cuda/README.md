# Dense cuBLAS benchmark

This bounded synthetic workload measures the CPU and cuBLAS dense evaluators,
including DLR primitives, ordinary/paired self-energy and single-particle APIs.
Inputs are identical for both backends; returned CUDA results are checked against CPU.

```bash
cmake --build cbuild-cuda --target dense_cuda_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild-cuda/benchmarks/dense_cuda/dense_cuda_benchmark 16 8 3 > measurements.csv
```

Load the environment matching the configured build. Arguments are Hilbert-space
dimension, maximum selected indices and warm repeats. CPU builds retain the host
measurements. Place collected CSV/profiling data under a hardware subfolder and
record the build revision, command, library versions and hardware alongside them.

[Workstation profiling evidence](thinkstationP620_RTXA4000/README.md) and
[comparison reports](../dense_comparison/README.md) are separate from portable code.
The optimization target is GH200-class hardware; local timings do not establish
performance or dispatch choices on another GPU.

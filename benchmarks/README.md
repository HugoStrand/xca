# Performance benchmarks

Dense evaluator benchmarks live in `dense_cpu`, `dense_cuda` (cuBLAS), and
`dense_cuda_small`. Shared comparisons and complete Solver workload controls live
in `dense_comparison`. Existing physics benchmarks in other directories are unchanged.

Configure with `-DBuild_Benchmarks=ON` to build the C++ benchmarks independently
of `Build_Tests` and `examples/`. By default this option follows `Build_Tests` on
first configuration. Binaries are placed in `<build>/benchmarks/<benchmark>/`.
These workloads are explicit performance runs, not automatic CTest timing tests.

Each benchmark directory contains portable sources and instructions. Recorded
measurements and hardware-specific interpretation belong in a hardware subfolder,
currently `thinkstationP620_RTXA4000`. For another machine, create a similarly
informative folder and preserve each run's manifest, raw samples and summary.
Do not infer a crossover on another GPU from workstation measurements. The
optimization target is high-end hardware such as GH200; RTX A4000 results are
local evidence.

Historical manifests retain their original commands, paths and source hashes.
They describe the measurement at its recorded revision, before this reorganization.
The experimental implementations and their results remain under `experiments/`.

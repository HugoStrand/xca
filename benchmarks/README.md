# Performance benchmarks

Dense workloads share fixtures and recording/comparison utilities in `dense_common`.
Backend entry points live in `dense_cpu`, `dense_cuda` (cuBLAS), and `dense_cuda_small`.
Other directories contain physics benchmarks.

Configure with `-DBuild_Benchmarks=ON` to build the C++ benchmarks independently
of `Build_Tests` and `examples/`. The default follows `Build_Tests` on first
configuration. Binaries are placed in `<build>/benchmarks/<backend>/`. Benchmarks
are explicit performance runs rather than automatic CTest timing tests.

Portable sources and instructions stay at each directory's root. Raw measurements,
profiling summaries and hardware-specific conclusions belong in hardware subfolders,
currently `thinkstationP620_RTXA4000`. For another machine, create an informative
folder and preserve each run's manifest, raw samples and summary. Recorded commands
and source hashes identify the code and environment used for each measurement.

The optimization target is high-end hardware such as GH200. Workstation measurements
provide local evidence and do not establish crossovers or dispatch choices on another
GPU. See [shared methodology](dense_common/README.md) before comparing results.

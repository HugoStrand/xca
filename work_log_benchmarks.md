# Benchmark reorganization work log

## 2026-10-13: layout and build fixtures

Scope: reorganize production dense benchmarks on `dev_cuda`; postpone history
squashing and leave `experiments/` unchanged. Preserve unrelated working-tree files.
The performance target remains GH200-class hardware, rather than the local RTX A4000.

- Moved all three benchmark C++ entry points out of `examples/` into `benchmarks/`.
- Added independent `Build_Benchmarks` configuration and per-backend CMake fixtures;
  native-small target is conditional on the native backend being available.
- Added `dense_cpu` entry point/runner using the unchanged shared physical harness.
  Moved original CPU/shared baseline evidence there, keeping mixed raw records intact.
- Moved comparison tools, Solver workload controls and cross-backend reports into
  `dense_comparison`. Portable instructions stay at each benchmark's first level.
- Grouped workstation measurements, profiling and interpretation under
  `thinkstationP620_RTXA4000`. Historical manifests retain their recorded provenance.
- Updated runner binary/source discovery and report links. Evaluator tests and
  experiments retain their existing shared-harness path.

Initial verification: Python syntax checks and CPU/comparison CLI help passed.
Build, runtime and data-preservation checks follow below before completion.

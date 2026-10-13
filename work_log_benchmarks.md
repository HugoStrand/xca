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

## 2026-10-13: verification and completion

Implementation commit: `8e950f5` (`[bench] reorganize dense benchmarks`).

- CPU and CUDA configurations/builds passed for the relocated targets. CPU
  configuration omits the unavailable native-small executable as intended.
- A separate configuration with `Build_Tests=OFF`, `PythonSupport=OFF`, and
  `Build_Benchmarks=ON` exposes all three CPU-compatible benchmark targets and
  no example targets; cached GoogleTest sources were reused without cloning.
- Dedicated CPU runner: 18 backend/case groups passed, all rows CPU-only.
  cuBLAS and native-small runners: 36 backend/case groups each passed for
  ordinary/paired SE and SPGF at orders 1/3. Matching comparison: 18 cases.
- Larger-matrix cuBLAS entry point ran successfully at dimension 4.
- Relocated Solver workload passed ordinary/dynamic controls against CPU for
  all three backends; subsequent machine-metadata recording also passed.
- Relevant CTests passed: CPU 2/2; CUDA 3/3, including native batch API tests.
- Every one of the 256 historical CSV/JSON/log records is preserved byte-for-byte.
  All benchmark Markdown links resolve; Python syntax and CLI checks passed.
- Future recorder manifests identify host/CPU/GPU and hash the relocated entry
  sources; the CPU wrapper is included in source provenance. Historical manifests
  remain unmodified. New smoke-run results/logs are under `/tmp/xca-bench-*`.

The initial checkout was on `dev_cuda_original`; this was caught at the first
checkpoint. Its prior tip `bf2ad3d` was restored, and all new commits are on
`dev_cuda`. History squashing is still postponed. Experiments, production evaluator
code, and unrelated user changes were left untouched.

## 2026-10-13: shared fixtures and public documentation

Implementation commit: `58d5543` (`[bench] share dense fixtures`).

- Renamed `dense_comparison` to `dense_common` and moved the physical fixture,
  workload/CSV contract, statistics recorder and comparisons there.
- Extracted cuBLAS reference construction and CUDA initialization, completion and
  profiling hooks to `dense_cuda/adapter.hpp`. CPU includes only the common harness.
  Native-small construction remains in its own entry point.
- Moved the cuBLAS physical-fixture entry to `dense_cuda_fixture_benchmark`; the
  small runner now defaults to the native executable. Each backend supplies its
  own runner defaults and source-provenance directories to the shared recorder.
- Retained a forwarding include at the existing experiment harness path, without
  changing experimental sources/results. Its factory-call interface compile-checks.
- Removed mandatory verification against a frozen development revision from the
  recorder. Current manifests hash shared/backend sources, including native kernels.
- Removed both ORIGINAL_README files. Portable methodology, argument semantics,
  numerical/timing rules and profiling instructions are in root READMEs.
  Workstation READMEs contain hardware, measured findings and limitations.
  Removed development chronology, supervisor progress and obsolete implementation
  acceptance claims from benchmark reports; raw provenance remains unchanged.

Verification: CPU/CUDA builds passed; focused CPU CTests 2/2 and CUDA CTests 3/3
passed (including native batch API). CPU runner passed 18 groups; cuBLAS and native
runners passed 36 groups each, with 18 matching comparison cases. The relocated
Solver control passed ordinary/dynamic comparisons for CPU/CUDA/native-small.
Benchmark-only CMake configuration exposes CPU and both cuBLAS workloads without
example/test targets. Python syntax/CLI checks and Markdown links pass. All 256
CSV/JSON/log records are preserved byte-for-byte, experiments are unchanged, and
no production evaluator or unrelated user files were edited. Verification logs and
smoke records are under `/tmp/xca-common-*`.

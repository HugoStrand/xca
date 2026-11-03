# ThinkStation P620 / Threadripper PRO 3945WX

Single-threaded CPU measurements on 2026-11-03, using Clang 19.1.1 Release
(`-O3`), system OpenBLAS, and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Dependencies: TRIQS `2930120`, NDA `4e59077`, fixed cppdlr `b9b24ab`.
CPU affinity was not pinned; boost and the `schedutil` governor were enabled.
Builds and other benchmark runs were kept separate from these measurements.
The workstation was not isolated from unrelated activity.

These are complete **prepared-backbone evaluator calls**, including discovery,
numerical work, joins, owning output and existing reset behavior. They exclude
backbone/model construction and are neither complete topology sums nor solver
iterations. See the [generic README](../README.md) for the exact workloads,
bounded selection procedure, validation and commands.

| Variant | Source checkpoint | Storage and arithmetic | Results |
| --- | --- | --- | --- |
| Baseline | `75a7c58` (sparse implementation `78576a6`) | Padded views, allocating vertex products, fixed cppdlr | [baseline.csv](baseline.csv) |
| Output GEMM | `6565624` | Padded views, reused vertex/endpoint outputs, active-only self-energy scaling, fixed cppdlr | [output_gemm.csv](output_gemm.csv) |

The baseline was compiled against frozen headers and static libraries from the
validated structural checkpoint. The same driver, model data and selected
backbones were used for comparison. All **56** cases passed elementwise checks
before and after repeated evaluation. Output GEMM matched the baseline exactly
on these samples. Its median productive-call speedup was 1.043x at rank 13 and
1.033x at rank 103; it is a small part of the intended convolution improvement.

[selected_backbones.csv](selected_backbones.csv) records each ordinal, discovered
path/segment counts and output magnitude. The sample intentionally separates
productive calls from zero-completion calls; it does not estimate their frequency
in a solver run. Three timing trials per row give median/minimum/maximum values.
Small differences should not be treated as portable speedups.

The local elementwise reference and statically linked baseline executable are
also saved in `cbuild/benchmarks/block_sparse_convolution/reference/` (ignored
build artifacts). Rebuilding the documented source checkpoint and rerunning
record mode regenerates a reference if these artifacts are cleaned.

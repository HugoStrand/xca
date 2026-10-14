# GH200 matched scalar density-set qualification

Job 3823469 completed exit 0 on n414, pinned source 6987c4d/GTest988ea2c, GCC14.3.0,
CUDA12.9.86, Arm CUDA TRIQS and native sm_90 sparse kernels. Five CTests passed
57.86 seconds, executing 38 active GoogleTests: multi-set18 and production
admission/device/evaluator/DLR3/5/8/4. No disabled/zero-test substitute was used.
New active/adjacent high-order and replay-heavy ownership controls passed on target.
Python, full-suite and full N=14 gates are separate.

Four workloads/four routes/three samples produced 48 rows; finite, CPU numerical
screens and exact repeatability passed. Maximum absolute error1.191e-16; GPU path
counts agree. Matched CPU shares the ordinary batch recipe/schedule, not the GPU
cutoff uncertainty-band replay. Rank23, one crossing topology, bounded indices.

| Workload | Original CPU ms | Matched CPU ms | General CUDA ms | Batched CUDA ms | Matched CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=4 NCA, two indices | 0.945 | 0.966 | 60.030 | 3.793 | 0.25 |
| N=4 OCA, eight indices | 26.122 | 4.575 | 358.998 | 7.663 | 0.60 |
| N=6 OCA, eight indices | 152.804 | 17.929 | 1973.367 | 19.528 | 0.92 |
| N=8 OCA, one index | 91.854 | 9.686 | 846.615 | 11.828 | 0.82 |

GPU improves materially over general CUDA but loses to matched CPU in every sampled
case. The large advantage over original CPU is substantially algorithmic. No
GPU-beats-matched-CPU, dispatch threshold or whole Solver speedup claim follows.
Next profile scoped full calls and investigate host descriptor/weight preparation,
transfers and synchronization before selecting a GPU policy. Three samples in one
allocation are limited crossover evidence; the fixed replay band is still not a
universal numerical guarantee.

One GPU allocated446 seconds including builds, nominal .12389 GPU-hours (billing1),
batch MaxRSS2,619,776 KiB. This investigation's four jobs total747 seconds, nominal
.2075 GPU-hours. The fifteen-minute cap held; no duplicate job was submitted.
Authentication expired before submission and was restored by the user. Current
projinfo reported2.87 GPU-hours/30 days before this job, with its reliability warning.

Future-dated archive source timestamps caused repeated builds. The isolated source
metadata was normalized to a fixed past date without content edits, avoiding more
rebuilds; the job recipe now prevents that issue. Full build logs and reusable Arm
tree remain at shared/qualification/xca-block-sparse-6987c4d. Shared checkouts/install
prefixes were untouched. Commands, source/binary/runner hashes, detailed tests,
modules/architecture/accounting and raw samples are retained here and in backend
folders. Setup/warm-up/CPU-screening calls remain outside reported API latency.

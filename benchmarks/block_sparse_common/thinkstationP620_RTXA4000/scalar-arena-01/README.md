# Contiguous scalar device storage

Code `ba6d0c8` (arena implementation `1e3d0c0`). All 30 isolated processes / 82 rows
passed numerical screening and exact replay; discovery counts match the previous
candidate. Source/binary/scheduler hashes and commands are in [manifest.json](manifest.json),
raw samples in [samples.csv](samples.csv), and medians in [summary.csv](summary.csv).
Profiler, build/test/MPI work was terminal before this unprofiled campaign.

CPU and arena were freshly paired; prior fan-in values come from scalar-fanin-01.
Eligibility/recipe/cutoff/order are unchanged. Scalar input/output tensors are
contiguous owned arenas, and block views remain borrowed only during owner lifetime.
Packing, transfer, result construction and all path work are inside call timing.
Other shapes/correlators retain general execution.

| Modes / partition / observable | CPU matched ms | GPU fan-in prior ms | GPU arena ms | CPU / arena |
| --- | ---: | ---: | ---: | ---: |
| 4 / occupation / se | 2.704 | 3.327 | 2.875 | 0.941 |
| 4 / number / se | 1.262 | 10.940 | 10.999 | 0.115 |
| 4 / mixed / se | 1.499 | 14.072 | 14.240 | 0.105 |
| 4 / occupation / spgf | 1.328 | 12.335 | 12.517 | 0.106 |
| 4 / occupation / correlator | 1.356 | 10.519 | 10.579 | 0.128 |
| 4 / number / spgf | 1.703 | 23.388 | 24.002 | 0.071 |
| 4 / number / correlator | 1.524 | 9.806 | 9.921 | 0.154 |
| 4 / mixed / spgf | 1.358 | 18.565 | 18.488 | 0.073 |
| 4 / mixed / correlator | 1.327 | 11.854 | 11.909 | 0.111 |
| 6 / occupation / se | 0.032 | 1.977 | 0.110 | 0.291 |
| 6 / number / se | 20.558 | 31.243 | 31.337 | 0.656 |
| 6 / mixed / se | 11.633 | 41.681 | 41.857 | 0.278 |
| 8 / number / se | 243.160 | 92.394 | 92.476 | 2.629 |
| 10 / number / se | 7184.116 | 699.485 | 697.401 | 10.301 |
| 10 / occupation / se | 22.992 | 56.143 | 25.330 | 0.908 |

The allocation mechanism is separately diagnosed in the
[scoped trace](../../../../experiments/block_sparse_cuda/scalar_paths/thinkstationP620_RTXA4000/profile-01/README.md):
3206 to 137 cudaMalloc calls, 2055 to 8 cudaMemcpy2D calls. Trace durations are
excluded from this table. Arena and control semantic suites pass 13 cases each;
five focused arena memory/leak cases pass with zero errors/leaked bytes.

Most rows have three samples, ten-mode rows one. A separate three-sample scalar
smoke is not merged into campaign medians. CPU batching only applies to eligible
scalar SE, with other rows using unchanged CPU. Near-parity or improvement over
old GPU does not meet a robust matched-CPU/target promotion threshold. GH200 remains
the target; no target default, whole Solver speedup or N=14 full workload is qualified.
All previous candidates/data remain. See [methodology](../../../../experiments/block_sparse_cuda/scalar_paths/README.md).

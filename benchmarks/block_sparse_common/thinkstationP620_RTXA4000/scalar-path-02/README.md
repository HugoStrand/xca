# Bounded scalar path candidate

Code checkpoint `629d086`. Clean corrected campaign: all 30 isolated processes /
82 rows pass CPU screening and exact replay. Discovery counts match production
native reference for every workload. The scalar worker optimizes self-energy for
1x1 blocks and singleton symmetry sets; other geometries/observables use the
retained general path. Raw [samples.csv](samples.csv), paired [summary.csv](summary.csv)
and source/binary/kernel/scheduler hashes in [manifest.json](manifest.json) are retained.

Native values are from prior clean ordered-production-01; CPU and candidate were
freshly paired here. CPU control is the hoisted evaluator, not a full batched scalar
CPU scheduler. No build/test/MPI job overlapped the corrected campaign.

| Modes / partition / observable | CPU hoisted ms | Native prior ms | Candidate ms | Native / candidate |
| --- | ---: | ---: | ---: | ---: |
| 4 / occupation / se | 0.555 | 15.007 | 3.477 | 4.316 |
| 4 / number / se | 0.503 | 10.907 | 10.902 | 1.000 |
| 4 / mixed / se | 0.535 | 14.124 | 14.079 | 1.003 |
| 4 / occupation / spgf | 0.409 | 12.450 | 12.347 | 1.008 |
| 4 / occupation / correlator | 0.511 | 10.534 | 10.476 | 1.006 |
| 4 / number / spgf | 0.766 | 23.762 | 23.289 | 1.020 |
| 4 / number / correlator | 0.591 | 9.759 | 9.863 | 0.989 |
| 4 / mixed / spgf | 0.541 | 18.263 | 18.257 | 1.000 |
| 4 / mixed / correlator | 0.550 | 11.962 | 11.811 | 1.013 |
| 6 / occupation / se | 0.768 | 2.018 | 2.020 | 0.999 |
| 6 / number / se | 7.039 | 31.614 | 31.098 | 1.017 |
| 6 / mixed / se | 3.761 | 42.075 | 41.606 | 1.011 |
| 8 / number / se | 135.152 | 92.875 | 93.154 | 0.997 |
| 10 / number / se | 4842.400 | 698.034 | 704.239 | 0.991 |
| 10 / occupation / se | 798.371 | 2845.913 | 90.745 | 31.362 |

The large scalar self-energy case spans multiple 512-lane cohorts and a tail,
retaining per-path output cutoff and original accumulation order. Separate six-mode
eight-index and ten-mode three-sample checks passed; the multi-cohort memcheck had
zero errors/leaked bytes. Eight inherited evaluator tests pass including changed
masks/range additivity and general fallback. Ten-mode rows in this campaign have
one repetition; the extra timing check is separate, not silently added to medians.

scalar-path-01 used a prior executable after a failed workspace-bound rebuild;
all its rows are excluded and retained with explicit mismatch notes. The corrected
candidate includes additional explicit workspace in admission/reported bounds.
Host metadata and opaque allocations remain outside full peak qualification.

This is selected-index full-call screening, not whole Solver throughput or a matched
batched-CPU solver comparison. Threshold-adjacent paths, larger symmetry sets,
correlator batching, dynamic addresses, budget/ownership limits and target promotion
remain pending. No Python/MPI selector exposes this private class. GH200 remains
the performance target; no target default or N=14 full workload is qualified.
See [methodology](../../../../experiments/block_sparse_cuda/scalar_paths/README.md).

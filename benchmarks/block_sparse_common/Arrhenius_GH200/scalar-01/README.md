# GH200 scalar-path batching qualification

Job 3819263 completed exit 0 on n439 with pinned source 3f462c3, GCC 14.3.0,
CUDA 12.9.86, Arm CUDA TRIQS and sm_90 native kernels. One GPU, 90 seconds allocated
including builds/tests: nominal 0.025 GPU-hours, billing TRES 1. The build reused the
initial qualified cache. Slurm's batch RSS sample was 36 MiB; this is not a reliable
whole-process/child peak. Backend CSV RSS high-water marks have their own scope.

All 14 active GoogleTests passed in 7.38 seconds, covering the reused eight evaluator
controls, cohort tails/changed masks, NCA cutoff-adjacent values, admission/empty
ranges, multi-field scalar fallback, ordered cancellation and coefficient updates.
LastTest.log retains individual names/results. Six workloads across four backends
produced 72 samples with finite outputs, CPU agreement and exact replay. Matched
CPU constants results equal the original CPU exactly. GPU structural counts match
the ordered control; maximum absolute CPU/GPU screening error was 1.955e-16.

Median complete synchronous call times (ms):

| Case | Original CPU | Ordered GPU | Batched CPU | Batched GPU | Ordered/batched GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scalar order 3, zero window | 0.048 | 0.550 | 0.008 | 0.093 | 5.93 |
| Scalar order 3, active window | 0.806 | 8.408 | 3.116 | 3.897 | 2.16 |
| N=6 scalar order 2, eight indices | 68.440 | 732.530 | 6.071 | 6.739 | 108.70 |
| N=10 scalar order 2, one index | 816.561 | 3362.550 | 18.371 | 17.587 | 191.19 |
| Scalar order 6, chain window | 0.373 | 0.546 | 0.015 | 0.095 | 5.73 |
| Mixed retarded correlator fallback | 1.045 | 10.802 | 1.038 | 10.793 | 1.00 |

The range accepts 640/3072 structural paths; N=10 accepts 2304/10240. Both zero
and order-six windows reject all paths: the latter does not qualify active high-order
arithmetic. Active order-three accepts 4/64; fallback accepts 10/36. Rank is 23.
The candidate materially reduces GPU scheduling/transfer overhead for larger scalar
cohorts, but GPU is slower than matched batched CPU except for a modest 1.045x N=10
ratio. Three samples in one allocation do not prove that small advantage is robust.
The candidate is restricted to singleton-field scalar self-energy; fallback timings
confirm general correlators still use the existing route. No production promotion
or automatic dispatch policy follows from this cohort.

Next close active high-order/cutoff, irregular scalar-set, lifecycle and broader
workload gates before promotion. Python/MPI and full N=14 remain separate gates.
Samples include discovery, packing, transfers and ordered fan-in; setup/warm-up
are separate. No full topology-sum or Solver throughput claim is made.
Commands, source/runner/binary hashes and raw samples are retained here and in the
matching CPU/CUDA folders. Full records/cache remain in
`shared/qualification/xca-block-sparse-3f462c3/logs/scalar-3819263` on Arrhenius.

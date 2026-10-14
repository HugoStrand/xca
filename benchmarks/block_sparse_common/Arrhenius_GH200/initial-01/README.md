# Initial matched GH200 cohort

Job 3818907 completed with exit 0 on n439, using pinned source 3f462c3, GCC 14.3.0,
CUDA 12.9.86 and native block-sparse sm_90 kernels. The same CUDA-enabled TRIQS prefix
was used for the unchanged CPU control and both CUDA frontends. One GPU was allocated
for 91 seconds including configuration/builds: nominally 0.02528 GPU-hours (billing
TRES 1). Batch MaxRSS was 1,062,976 KiB. No extra CPU/memory resources were requested;
the allocation supplied 72 CPUs, with build concurrency capped at six.

Seven cases, three backends and three samples yielded 63 rows. Each CUDA process
passed the built-in rotated-CPU numerical screen and exact repeated-call check.
Maximum observed absolute error was 2.865e-14. Median synchronous API times follow;
setup, warm-up and the separate CPU screening call are excluded from these medians.

| Case | CPU ms | Library CUDA ms | Ordered CUDA ms | CPU/ordered | Accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scalar SE, first=0 | 0.049 | 0.538 | 0.548 | 0.09 | 0/64 |
| Scalar SE, first=1 | 0.815 | 15.099 | 8.511 | 0.10 | 4/64 |
| Mixed retarded correlator | 1.059 | 18.155 | 11.039 | 0.10 | 10/36 |
| N=6 number retarded SPGF | 22.754 | 141.426 | 67.671 | 0.34 | 180/252 |
| N=6 number SE, order 3 | 27.965 | 68.535 | 26.392 | 1.06 | 4/7 |
| N=8 number SE, NCA | 98.369 | 192.372 | 67.557 | 1.46 | 8/9 |
| N=10 number SE, NCA | 4827.384 | 415.902 | 169.533 | 28.47 | 10/11 |

The first scalar window is zero-survival and has zero signal; its latency cannot
measure arithmetic throughput. The next scalar window is active. N=6 order-three
SE has a weak signal (3.54e-8). All cases use rank 23, one indexed evaluation and
one crossing topology. These are complete calls, not whole topology sums or Solver
iterations. Three samples from one allocation do not determine statistical dispatch
thresholds. The retained CPU algorithm is unchanged, not transform-hoisted to match
CUDA. RSS includes setup and controls; sampled device-free-memory deltas are not
continuous peaks.

Ordered CUDA is promising for larger number sectors but remains slower than CPU on
small blocks. Next qualify the retained scalar-path batching experiment on GH200,
with active/rejected windows and semantic controls, before considering promotion.
No automatic production backend policy changes follow from this cohort.

`manifest.json` records commands, revisions, runner/binary hashes and process results.
`samples.csv` and `summary.csv` retain raw and median timing evidence; backend CSVs
are in the CPU/CUDA benchmark folders under the same machine/campaign name.
Full build/hardware records and the reusable Arm tree remain at
`shared/qualification/xca-block-sparse-3f462c3/logs/cohort-3818907` on Arrhenius.

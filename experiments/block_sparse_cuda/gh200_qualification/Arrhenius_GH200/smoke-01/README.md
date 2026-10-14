# First GH200 C++ qualification

Job 3818566 completed with exit 0 on node n439: one GH200, 72 allocated CPUs,
compiler concurrency capped at six. Elapsed allocation was 120 seconds, nominally
0.0333 GPU-hours; Slurm reports billing TRES 1. This is allocated GPU time, not a
claim about the project's precise monthly accounting. The batch MaxRSS was 828 MiB.

Pinned source 3f462c3 and GTest 988ea2c were transferred as source-only archives;
the shared checkout and installs were unchanged. GCC 14.3.0, CUDA toolkit 12.9.86,
Arm CUDA TRIQS 4.0 and native block-sparse kernels targeting sm_90 were used.
Python, dense-small native kernels and benchmarks were disabled.

Four CTests passed in 10.63 seconds, executing 20 GoogleTests: admission (3),
device operations (5), evaluator (8), and DLR operations (4). LastTest.log preserves
individual test names and numerical controls. No disabled/zero-test CUDA substitute
was used. This covers focused SE/SPGF/correlator controls and basic lifetime/admission
checks, not the complete suite, Python/MPI, target performance or full N=14 workloads.

The Arm build cache and full configure/build/hardware records remain at
`/nobackup/proj/disk/naiss2026-3-569/shared/qualification/xca-block-sparse-3f462c3`.
Reuse that cache for the next bounded cohort; preserve its source revision.

# Threaded CPU number-sector control

Threadripper PRO3945WX (12physical/24logical cores), CPU TRIQS/OpenBLAS0.3.26 Zen
pthread build. One OMP thread, OpenBLAS1versus4threads. Linux process status sampled
outside the benchmark confirms peaks1/4 respectively. Fixed N10 total-number NCA,
rank23, max sector252, one topology/index, three fresh processes per thread count,
one warm call and three timed calls per process. All six processes pass the runner's
within-process repeatability/finite screen. Raw signal norms agree across thread counts;
this alone is not a full-output cross-thread bitwise proof.

Median of process medians: one thread3798.876ms, four threads2834.873ms
(1.34x). Full process medians, setup/RSS/samples and exact commands are
retained. CPU/GPU controls must use matched scientific inputs; this is a bounded
larger-matrix CPU baseline, not a comparison to an unmeasured GPU counterpart.
It establishes no optimal thread count or GH200 CPU/GPU crossover. Target remains
GH200-class, and this workstation supplies development evidence only.

Binary/shared fixture/source/library hashes, CPU affinity and observed worker PIDs
are retained. Source checkpoint2713bb0/runtime85ecb2d; no evaluator changes. No GPU
or Arrhenius allocation. Nominal GPU accounting is unchanged by these CPU-only runs.

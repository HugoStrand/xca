# ThinkStation P620 / RTX A4000 reference calls

Performance target: GH200-class hardware. These local measurements screen the
retained synchronous reference and do not set target dispatch defaults.

Campaign `reference-01` ran 30 isolated processes successfully, yielding 82 raw
samples from 15 paired workloads. Code checkpoint: `d79e8d7`. Source and binary
hashes, installed TRIQS headers, commands and machine metadata are in
[manifest.json](reference-01/manifest.json); [samples.csv](reference-01/samples.csv)
retains individual repetitions and [summary.csv](reference-01/summary.csv) pairs
median complete-call latencies. CPU/GPU raw rows live in their respective benchmark
hardware directories. Timing excludes constructor/model preparation; see the
[shared definitions](../README.md) for what each counter measures.

Each case uses the first indexed evaluation of one crossing topology. Modes 4/6/8
have three measured repetitions; modes 10 have one, so those are observations,
not a noise-qualified promotion decision. All are single-thread CPU controls;
the unchanged CPU convolution is not yet the algorithm-matched hoisted control.
All CUDA screening comparisons and exact repeated-output checks passed.

| Modes / partition / observable | CPU ms | CUDA ms | CPU / CUDA |
| --- | ---: | ---: | ---: |
| 4 / occupation / se | 1.810 | 29.109 | 0.062 |
| 4 / number / se | 1.236 | 32.511 | 0.038 |
| 4 / mixed / se | 1.476 | 32.233 | 0.046 |
| 4 / occupation / spgf | 1.232 | 23.899 | 0.052 |
| 4 / occupation / correlator | 1.322 | 19.364 | 0.068 |
| 4 / number / spgf | 1.663 | 56.313 | 0.030 |
| 4 / number / correlator | 1.493 | 18.511 | 0.081 |
| 4 / mixed / spgf | 1.338 | 40.308 | 0.033 |
| 4 / mixed / correlator | 1.311 | 22.466 | 0.058 |
| 6 / occupation / se | 0.741 | 1.959 | 0.378 |
| 6 / number / se | 20.374 | 91.322 | 0.223 |
| 6 / mixed / se | 11.528 | 93.582 | 0.123 |
| 8 / number / se | 244.829 | 265.295 | 0.923 |
| 10 / number / se | 7008.667 | 1142.758 | 6.133 |
| 10 / occupation / se | 1021.276 | 5603.625 | 0.182 |

The six-mode occupation order-three window has zero signal and no accepted paths;
retain it as a pruning/orchestration control. Ten-mode number sectors reach maximum
block dimension 252; CUDA needs about 1.14 s versus 7.01 s for the unchanged CPU.
Ten-mode occupation sectors remain scalar and GPU launch/library overhead is much
larger than CPU latency. This supports investigating the two regimes separately;
it does not establish full topology-sum or Solver speedups. DLR algorithm-matched
CPU controls, detailed launch/transfer profiling, changed-input replay and target
hardware measurements remain outstanding. No N=14 diagonalization was performed.

## Matched CPU convolution control

[hoisted-cpu-02](hoisted-cpu-02/README.md) supplies the matched full-evaluator CPU
control and fresh paired CUDA timings. Its CPU outputs match the original bit for
bit. Use those ratios when assessing device advantage. `hoisted-cpu-01` is excluded
from timing comparisons because it overlapped a verification build.

[Ordered sparse candidate complete calls](ordered-evaluator-01/README.md) retain the original GPU
reference and compare an isolated native operator route. These are local screening
results; production Python/MPI promotion and target qualification remain pending.

[Production ordered route timings](ordered-production-01/README.md) measure the explicitly selected
integrated backend. The private candidate and library baseline remain separate.

[Bounded scalar self-energy scheduler](scalar-path-02/README.md) retains general
execution for other cohorts. scalar-path-01 is excluded because the revised
workspace-bound source had not rebuilt; corrected data is in scalar-path-02.

[Matched scalar CPU/GPU scheduling](scalar-matched-01/README.md) isolates algorithmic
batching gains. The previous hoisted CPU comparison is not evidence that scalar
GPU scheduling beats matched CPU; this full-call screen favors CPU in that regime.

[Output-owned scalar GPU accumulation](scalar-fanin-01/README.md) improves the
serial candidate while retaining original contribution order. The measured scalar
GPU still trails matched CPU; no production promotion follows from this screen.

[Contiguous scalar device arenas](scalar-arena-01/README.md) reduce allocation/copy
multiplicity identified by a separately scoped trace. Unprofiled matched timings
remain separate from profiler diagnostics; all prior candidates are retained.

[Path-constant scalar coefficient preparation](scalar-constants-01/README.md)
compares matched CPU and broadcast GPU variants, including fresh five-sample scalar
controls. Local near-parity does not select production or GH200 dispatch.

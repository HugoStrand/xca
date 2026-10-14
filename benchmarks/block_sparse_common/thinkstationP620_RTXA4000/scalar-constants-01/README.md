# Scalar path-constant coefficients

Implementation `6068a8b`. The new GPU variant computes path-constant vertex,
endpoint, prefactor and sign coefficients once per lane and uploads only one
coefficient per lane for a broadcast multiply. Node-dependent weights retain their
full representation. The matched CPU variant uses the same once-per-lane host work.
Operation order, scalar eligibility, cutoff and ordered fan-in are unchanged;
other shapes and observables retain general execution. Production dispatch is unchanged.

All 30 isolated processes / 82 samples passed; every CPU result matched the original
CPU evaluator bit for bit. GPU results repeat exactly, and discovery counts match
scalar-arena-01. Numerical screening is separate from the stricter exact-input
qualification suite. Fourteen new-variant tests and thirteen retained arena-control
tests passed. Six focused tests passed memcheck with zero errors and zero leaked bytes.
The CPU control also built and passed an eight-index, multi-cohort case without CUDA.
Builds, tests and sanitizer runs were terminal before timing.

[Manifest](manifest.json) records commands, source/header/binary hashes and hardware.
[Raw samples](samples.csv) and [summary](summary.csv) preserve the standard bounded
campaign. Prior arena timings are from a separate campaign, not interleaved controls.
Most cases have three samples; ten-mode cases have one. Only eligible scalar SE
uses matched CPU batching; other rows retain unchanged CPU/general GPU evaluation.

| Modes / partition / observable | CPU constants ms | GPU constants ms | GPU arena prior ms |
| --- | ---: | ---: | ---: |
| 4 / occupation / se | 2.729 | 2.650 | 2.875 |
| 4 / number / se | 1.252 | 10.805 | 10.999 |
| 4 / mixed / se | 1.460 | 14.197 | 14.240 |
| 4 / occupation / spgf | 1.273 | 12.393 | 12.517 |
| 4 / occupation / correlator | 1.353 | 10.568 | 10.579 |
| 4 / number / spgf | 1.655 | 23.454 | 24.002 |
| 4 / number / correlator | 1.515 | 9.851 | 9.921 |
| 4 / mixed / spgf | 1.340 | 18.451 | 18.488 |
| 4 / mixed / correlator | 1.334 | 11.900 | 11.909 |
| 6 / occupation / se | 0.032 | 0.112 | 0.110 |
| 6 / number / se | 20.237 | 31.160 | 31.337 |
| 6 / mixed / se | 11.532 | 41.783 | 41.857 |
| 8 / number / se | 241.355 | 93.256 | 92.476 |
| 10 / number / se | 7215.379 | 698.288 | 697.401 |
| 10 / occupation / se | 20.744 | 21.944 | 25.330 |

A separate fresh scalar comparison runs five samples per case for both CPU variants
and both GPU variants, after building the retained CPU control. Its
[manifest](repeated-scalar/manifest.json) pins commands and executable hashes; raw
CSV files and [summary](repeated-scalar/summary.csv) remain separate from campaign
medians. There is one warm-up, all five timed repetitions in one process, per case;
this is not five independent process repetitions.

| Modes | CPU baseline ms | CPU constants ms | GPU arena ms | GPU constants ms |
| --- | ---: | ---: | ---: | ---: |
| 4 | 2.707 | 2.687 | 2.817 | 2.636 |
| 10 | 22.398 | 20.993 | 24.898 | 21.179 |

Ten-mode GPU latency falls about 14.9% against the freshly measured arena control;
the matched CPU also improves about 6.3%. GPU and matched CPU remain essentially at
parity on this local selection. This does not meet a robust 15% promotion gate or
establish a GPU advantage. No tolerance/default changes or full Solver throughput
claims follow. Maximum GPU campaign screening error is 3.645e-13 in the existing
weak higher-order non-scalar case; ten-mode scalar error is 4.926e-15.

GH200-class hardware remains the target. This A4000 experiment does not qualify
GH200 dispatch, non-singleton symmetry-set batching, optimized correlators, complete
host/opaque memory admission, or a full fourteen-mode evaluation. All prior controls
and experiments remain. See [methodology](../../../../experiments/block_sparse_cuda/scalar_paths/README.md).

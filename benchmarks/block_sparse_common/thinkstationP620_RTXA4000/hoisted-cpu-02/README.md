# Matched CPU / CUDA complete calls

Code checkpoint `6446c2b`, generated-control code `79cb24c`. Clean campaign
`hoisted-cpu-02` ran after verification builds and other benchmarks terminated.
All 30 processes and 82 samples passed. All 15 hoisted CPU cases matched the
unchanged CPU control bit for bit, including ordinary/retarded models, every
observable and a zero-survival window. CUDA screening checks also passed.

Source/binary/header hashes and commands are in [manifest.json](manifest.json),
with raw [samples.csv](samples.csv) and paired [summary.csv](summary.csv). The
summary also includes the unchanged CPU latencies from `reference-01`, which is
a separate run. CPU-control timing within this campaign includes output flattening
and must not substitute for the separately measured original baseline.

| Modes / partition / observable | Hoisted CPU ms | CUDA ms | Hoisted CPU / CUDA |
| --- | ---: | ---: | ---: |
| 4 / occupation / se | 0.563 | 29.373 | 0.019 |
| 4 / number / se | 0.503 | 32.835 | 0.015 |
| 4 / mixed / se | 0.528 | 32.737 | 0.016 |
| 4 / occupation / spgf | 0.414 | 24.092 | 0.017 |
| 4 / occupation / correlator | 0.505 | 19.478 | 0.026 |
| 4 / number / spgf | 0.761 | 56.307 | 0.014 |
| 4 / number / correlator | 0.603 | 18.488 | 0.033 |
| 4 / mixed / spgf | 0.551 | 40.178 | 0.014 |
| 4 / mixed / correlator | 0.547 | 22.513 | 0.024 |
| 6 / occupation / se | 0.740 | 1.947 | 0.380 |
| 6 / number / se | 6.939 | 91.278 | 0.076 |
| 6 / mixed / se | 3.784 | 92.332 | 0.041 |
| 8 / number / se | 131.301 | 266.789 | 0.492 |
| 10 / number / se | 5039.565 | 1130.458 | 4.458 |
| 10 / occupation / se | 794.695 | 5614.407 | 0.142 |

GH200-class accelerators remain the target. Local matched comparisons materially
change the eight-mode conclusion: hoisted CPU is faster there. CUDA retains a
larger-sector advantage in the single sampled ten-mode number window. Small/scalar
sectors still need launch/packing/DLR amortization; these results do not choose a
sparse-product kernel or production dispatch. Modes 10 have one repetition and
cannot establish promotion/noise gates. No full topology sum, Solver speedup or
N=14 production workload is qualified.

`hoisted-cpu-01` overlapped CPU-only compilation; its numerical checks are retained
but **all timing rows are excluded**. Primitive controls and the private generated
adapter are in [the isolated experiment](../../../../experiments/block_sparse_cuda/dlr_hoisted/README.md).
Original and CUDA evaluators remain unchanged. A threaded CPU comparison, phase
profiling, changed-propagator calls and target-hardware measurements remain pending.

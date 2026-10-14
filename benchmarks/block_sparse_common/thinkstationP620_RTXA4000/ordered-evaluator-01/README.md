# Ordered sparse kernel: complete evaluator calls

Performance target: GH200-class hardware. Code checkpoint `1cc785c`. The private
candidate changes sparse operator application only. All 30 isolated processes /
82 samples passed CPU screening and exact repeated-output checks. Matched CPU
outputs remain bitwise identical to unchanged CPU. Discovery counts agree with the
retained GPU reference in every case, including the zero-survival window.

[manifest.json](manifest.json) pins source/CUDA kernel/binary/header hashes and
commands; [samples.csv](samples.csv) retains raw repetitions/errors and
[summary.csv](summary.csv) pairs medians. Library reference values come from the
separate clean `hoisted-cpu-02` run; CPU and candidate CUDA were freshly paired here.
This is a fixed-input, selected-index comparison, not a whole Solver benchmark.

| Modes / partition / observable | CPU hoisted ms | GPU library ms (prior) | GPU ordered ms | Library / ordered |
| --- | ---: | ---: | ---: | ---: |
| 4 / occupation / se | 0.574 | 29.373 | 14.347 | 2.047 |
| 4 / number / se | 0.498 | 32.835 | 10.113 | 3.247 |
| 4 / mixed / se | 0.538 | 32.737 | 13.473 | 2.430 |
| 4 / occupation / spgf | 0.415 | 24.092 | 11.811 | 2.040 |
| 4 / occupation / correlator | 0.511 | 19.478 | 10.144 | 1.920 |
| 4 / number / spgf | 0.772 | 56.307 | 22.489 | 2.504 |
| 4 / number / correlator | 0.615 | 18.488 | 9.395 | 1.968 |
| 4 / mixed / spgf | 0.545 | 40.178 | 17.586 | 2.285 |
| 4 / mixed / correlator | 0.555 | 22.513 | 11.403 | 1.974 |
| 6 / occupation / se | 0.755 | 1.947 | 1.965 | 0.991 |
| 6 / number / se | 7.021 | 91.278 | 28.984 | 3.149 |
| 6 / mixed / se | 3.781 | 92.332 | 39.726 | 2.324 |
| 8 / number / se | 135.243 | 266.789 | 85.673 | 3.114 |
| 10 / number / se | 5064.838 | 1130.458 | 690.846 | 1.636 |
| 10 / occupation / se | 788.104 | 5614.407 | 2710.869 | 2.071 |

Most cases have three measured repetitions; ten-mode cases have one and cannot
establish noise-qualified promotion there. The zero-survival six-mode occupation
window is retained. Scalar ten-mode CUDA remains slower than matched CPU despite
its improvement, supporting additional bounded batching/DLR amortization work.
Numerical screening is supplemented by ten inherited/snapshot C++ tests and four
focused memcheck tests (zero errors / zero bytes leaked). The production Python/MPI
backend is unchanged; candidate promotion still requires those integration gates.

No N=14 full evaluation or multi-device switch was qualified. The operator primitive
sweep separately covers exact 3003-by-3432 geometry with narrow RHS slabs. Target
hardware, wider RHS, changed-input/operator replay and complete peak qualification
remain pending. The retained reference stays available. See the
[private candidate methodology](../../../../experiments/block_sparse_cuda/operator_apply/README.md).

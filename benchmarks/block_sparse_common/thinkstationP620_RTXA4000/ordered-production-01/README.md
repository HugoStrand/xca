# Production ordered operator route

Performance target: GH200-class hardware. Code checkpoint `5d0a825`, integration
`938e73b`. All 30 isolated processes / 82 samples passed CPU screening and exact
replay. Matched CPU outputs remain bitwise identical to original CPU; discovery
counts match the retained GPU library reference. No build/test/MPI job overlapped
this campaign. [manifest.json](manifest.json) pins commands, source/native-kernel/
binary/header hashes; [samples.csv](samples.csv) retains repetitions, numerical
errors and setup/call counters; [summary.csv](summary.csv) pairs medians.

The production snapshot still constructs CSR descriptors and uses its original
ownership/budget/device validation, so this measures the actual selected route,
not the private prototype. Library values are from the prior clean `hoisted-cpu-02`
run; matched CPU and production native were freshly paired here.

| Modes / partition / observable | CPU matched ms | GPU library ms (prior) | GPU ordered ms | Library / ordered |
| --- | ---: | ---: | ---: | ---: |
| 4 / occupation / se | 0.569 | 29.373 | 15.007 | 1.957 |
| 4 / number / se | 0.506 | 32.835 | 10.907 | 3.010 |
| 4 / mixed / se | 0.530 | 32.737 | 14.124 | 2.318 |
| 4 / occupation / spgf | 0.415 | 24.092 | 12.450 | 1.935 |
| 4 / occupation / correlator | 0.494 | 19.478 | 10.534 | 1.849 |
| 4 / number / spgf | 0.776 | 56.307 | 23.762 | 2.370 |
| 4 / number / correlator | 0.604 | 18.488 | 9.759 | 1.894 |
| 4 / mixed / spgf | 0.549 | 40.178 | 18.263 | 2.200 |
| 4 / mixed / correlator | 0.545 | 22.513 | 11.962 | 1.882 |
| 6 / occupation / se | 0.758 | 1.947 | 2.018 | 0.965 |
| 6 / number / se | 7.004 | 91.278 | 31.614 | 2.887 |
| 6 / mixed / se | 3.752 | 92.332 | 42.075 | 2.194 |
| 8 / number / se | 130.203 | 266.789 | 92.875 | 2.873 |
| 10 / number / se | 5091.182 | 1130.458 | 698.034 | 1.619 |
| 10 / occupation / se | 802.016 | 5614.407 | 2845.913 | 1.973 |

Most cases have three repetitions, but ten-mode cases have one; do not use them to
set a noise-qualified target dispatch threshold. The zero-survival window is kept.
Full CPU/CUDA/native-disabled suites and two-rank all-observable/error recovery
checks qualify the integrated selector; these are still bounded-index timings,
not complete Solver/topology-sum performance. Scalar sectors continue to need
batching/DLR amortization. Wider RHS, complete peak admission and GH200 measurements
remain pending. Library execution remains the default and all experiments are retained.

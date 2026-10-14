# Matched scalar CPU / GPU full calls

Code `1427e2a`. Both controls share recipe.inc for stage order and scalar path
scheduling. CPU uses cached LU, packed rank transforms and original ordered fan-in;
GPU uses the same recipe with owned device work. Eligibility is all-1x1 blocks and
singleton symmetry sets for self-energy. Other shapes and observables fall back to
unchanged CPU/general GPU, so those rows are not a batched CPU comparison.

All 30 isolated processes / 82 rows passed numerical screening and exact replay.
[manifest.json](manifest.json) pins source/recipe/binary/header fingerprints and
commands; [samples.csv](samples.csv) stores repetitions/errors; [summary.csv](summary.csv)
pairs medians. No builds/tests overlapped the paired campaign. The CPU control also
builds/runs in the CPU-only tree, and extraction of the shared recipe left the GPU
12-test qualification passing.

| Modes / partition / observable | CPU ms | GPU ms | CPU / GPU |
| --- | ---: | ---: | ---: |
| 4 / occupation / se | 2.723 | 3.429 | 0.794 |
| 4 / number / se | 1.263 | 10.807 | 0.117 |
| 4 / mixed / se | 1.495 | 14.067 | 0.106 |
| 4 / occupation / spgf | 1.297 | 12.557 | 0.103 |
| 4 / occupation / correlator | 1.369 | 10.584 | 0.129 |
| 4 / number / spgf | 1.695 | 23.751 | 0.071 |
| 4 / number / correlator | 1.524 | 9.830 | 0.155 |
| 4 / mixed / spgf | 1.359 | 18.260 | 0.074 |
| 4 / mixed / correlator | 1.335 | 11.855 | 0.113 |
| 6 / occupation / se | 0.032 | 1.981 | 0.016 |
| 6 / number / se | 20.366 | 31.187 | 0.653 |
| 6 / mixed / se | 11.438 | 42.219 | 0.271 |
| 8 / number / se | 237.941 | 92.570 | 2.570 |
| 10 / number / se | 6967.092 | 701.660 | 9.929 |
| 10 / occupation / se | 23.262 | 89.399 | 0.260 |

The selected ten-mode scalar self-energy window favors matched CPU, unlike the
comparison against the older hoisted evaluator. Most earlier scalar gains were
algorithmic. GPU ordered per-path cutoff/fan-in still uses serialized host/device
handoffs; investigate that cost before claiming a useful scalar route over matched
CPU. The zero-survival window is retained and also favors CPU.

Most cases have three repetitions; ten-mode points have one. A separate three-sample
CPU validation was run under concurrent build activity and is not timing evidence.
This screen does not choose a production/GH200 default or prove whole Solver speedup.
Both schedulers keep the original per-path cutoff and accumulation order. Matched
CPU has no device memory counters/count observer and reports NA path columns.
GH200-class hardware remains the target; neither target hardware nor N=14 full
workloads are qualified. See [methodology](../../../../experiments/block_sparse_cuda/scalar_paths/README.md).

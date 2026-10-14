# ThinkStation P620 / RTX A4000 scalar edge batches

Performance target: GH200-class hardware. Code checkpoint and exact hashes are in
[manifest.json](scalar-01/manifest.json). All 15 isolated cases / 135 samples passed
per-lane CPU screening and exact replay. Inputs include zero and weak lanes; no
batch-wide norm is used. Width-17 Lambda=120 memcheck reported zero errors and
zero leaked allocations. A CUDA build and CPU-only configure passed.

This is an independent-edge segment, not diagram/Solver throughput. GPU input
uploads, allocator/context setup and host output download are outside timing.
Inclusive packing/scattering uses per-lane device-to-device copies. Resident input
packing is excluded explicitly; output scattering remains. CPU supplies numerical
controls, not a timed matched CPU batch in this first screen. Three raw repetitions
per case are in [samples.csv](scalar-01/samples.csv); medians and ratios are in
[summary.csv](scalar-01/summary.csv).

| Lambda | Rank | Lanes | Independent ms | Packed inclusive ms | Packed resident ms | Inclusive ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 23 | 1 | 0.3101 | 0.2790 | 0.2690 | 1.11 |
| 40 | 23 | 2 | 0.6165 | 0.3359 | 0.3183 | 1.84 |
| 40 | 23 | 4 | 1.2175 | 0.3694 | 0.3424 | 3.30 |
| 40 | 23 | 8 | 2.4950 | 0.5303 | 0.4787 | 4.71 |
| 40 | 23 | 17 | 5.2211 | 0.6061 | 0.5061 | 8.61 |
| 40 | 23 | 32 | 9.6825 | 0.7226 | 0.5451 | 13.40 |
| 40 | 23 | 64 | 19.4108 | 0.9924 | 0.6381 | 19.56 |
| 40 | 23 | 127 | 38.5839 | 1.5203 | 0.8091 | 25.38 |
| 40 | 23 | 256 | 77.6727 | 2.6272 | 1.1995 | 29.57 |
| 10 | 15 | 1 | 0.2910 | 0.2595 | 0.2504 | 1.12 |
| 10 | 15 | 17 | 4.8562 | 0.4749 | 0.3712 | 10.23 |
| 10 | 15 | 127 | 35.8473 | 1.3631 | 0.6695 | 26.30 |
| 120 | 32 | 1 | 0.3118 | 0.2906 | 0.2797 | 1.07 |
| 120 | 32 | 17 | 5.3535 | 0.6547 | 0.5507 | 8.18 |
| 120 | 32 | 127 | 39.0440 | 1.6119 | 0.8955 | 24.22 |

Packing/solve amortization improves larger cohorts; singleton results reveal
fixed overhead and must remain in dispatch evidence. This does not select a GPU
batch size/default on target hardware. A useful evaluator queue must keep
individual completed-path cutoffs and original accumulation order, copy borrowed
backbone metadata before retaining it, bound ready-work memory and handle irregular
lengths/zero-survival tails. No path/topology reassociation or cutoff was implemented
here. Timed CPU packing/solve controls and a complete candidate evaluator remain
required before promotion. Wider matrix blocks need a distinct rectangular plan.
See [methodology](../README.md). Existing implementations/experiments are retained.

# ThinkStation P620 / RTX A4000 DLR controls

GH200-class accelerators remain the performance target. This local E5 control
qualifies hoisting on bounded primitive shapes; it does not promote a production
route or establish complete-diagram speedup.

Code checkpoint `d8f3c44`; eight isolated cases and all 108 samples passed. Both
CPU routes matched bit for bit for coefficient convolution and the complete edge
(two coefficient solves plus convolution). All repeated outputs were exact and
CUDA screening checks passed. Source/binary/cppdlr hashes and commands are in
[manifest.json](hoisted-01/manifest.json). Raw [samples.csv](hoisted-01/samples.csv)
and paired [summary.csv](hoisted-01/summary.csv) retain numerical errors.

| Lambda | Shape M/K/N | Stage | CPU original ms | CPU hoisted ms | CUDA ms | Hoisted / CUDA |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 40 | 1/1/1 | coefficients | 0.0639 | 0.0120 | 0.1373 | 0.087 |
| 40 | 1/1/1 | edge | 0.0659 | 0.0144 | 0.3218 | 0.045 |
| 40 | 4/4/4 | coefficients | 0.1348 | 0.0261 | 0.2812 | 0.093 |
| 40 | 4/4/4 | edge | 0.1528 | 0.0404 | 0.5527 | 0.073 |
| 40 | 4/6/3 | coefficients | 0.1569 | 0.0262 | 0.2782 | 0.094 |
| 40 | 4/6/3 | edge | 0.1774 | 0.0439 | 0.5732 | 0.077 |
| 40 | 10/20/10 | coefficients | 1.0318 | 0.1205 | 0.3274 | 0.368 |
| 40 | 10/20/10 | edge | 1.1914 | 0.2808 | 0.6058 | 0.463 |
| 40 | 56/70/56 | coefficients | 47.7103 | 3.9475 | 1.7481 | 2.258 |
| 40 | 56/70/56 | edge | 50.8498 | 7.4268 | 2.9263 | 2.538 |
| 40 | 210/252/210 | coefficients | 822.2752 | 145.7357 | 43.4309 | 3.356 |
| 40 | 210/252/210 | edge | 894.0874 | 209.5566 | 56.4918 | 3.710 |
| 10 | 4/6/3 | coefficients | 0.0607 | 0.0155 | 0.2290 | 0.068 |
| 10 | 4/6/3 | edge | 0.0698 | 0.0266 | 0.4437 | 0.060 |
| 120 | 4/6/3 | coefficients | 0.3642 | 0.0391 | 0.2876 | 0.136 |
| 120 | 4/6/3 | edge | 0.3887 | 0.0633 | 0.6206 | 0.102 |

CPU is single-threaded. Most cases have three repetitions; the largest geometry
and the additional Lambda=10/120 controls have one. Initial uploads, context/handle
setup and sparse vertices are excluded; CUDA output download and synchronization
are included, and device scratch is retained. These timing scopes are primitive
controls and differ from complete evaluator calls. See [methodology](../README.md).
A full-evaluator hoisted CPU control, changed-input measurements, sensible threaded
CPU control and target-hardware runs remain pending. No N=14 diagonalization or
full diagram was performed.

# GH200 real NCA qualification

Pinned source b5a131c, recipe 0a03ade; hashes, commands and binary identities are
in manifest.json. Arrhenius n407, NVIDIA GH200, GCC 14.3 / CUDA 12.9.86.
The reported product is GH200 120GB; observed FB memory is 97871 MiB (95.58 GiB).
Single OpenMP/BLAS thread; one selected topology/index, rank 23, three warm samples.

Job 3843405 completed: three CTests / 20 active GoogleTests, then four cases,
four routes and 48 samples. All processes passed finite/numerical/repeatability
screens. GPU signal/error/path-count summaries agree offline. The CPU constants
control uses different preparation from the unchanged CPU and has no path counts.

Median synchronized C++ call times (ms):

| Case | CPU constants | General ordered | Prototype | Production |
| --- | ---: | ---: | ---: | ---: |
| N4 occupation NCA | .186234 | 10.028712 | 2.240698 | .238457 |
| N8 occupation NCA | 3.624880 | 292.954774 | 3.167613 | 1.399253 |
| N4 OCA fallback | 1.720394 | 14.896593 | 14.107722 | 14.142441 |
| N4 number fallback | .199994 | 9.148611 | 9.103429 | 9.119172 |

These samples qualify the guarded NCA route and fallback behavior on target hardware;
they establish no general crossover, default selection or whole Solver speedup.
No target Python/MPI, full suite, physical N10/N14 or row-convolution run is claimed.
The observed memory cannot accommodate the current N14/rank64 116.921 GiB NCA or
184.318 GiB convolution device bounds; lower-rank diagnostics are not physical proof.

Job 3843116 was cancelled after preparation stalled on filesystem timestamp updates;
those updates eventually completed and partial compiled output was retained. There
was no numerical/compiler failure. Timestamp normalization now occurs before transfer,
with an epoch marker checked before configuration. The retry reused the isolated build.
Accounting includes both allocations: 392 + 182 = 574 GPU-seconds (.15944 GPU-hours).
Tracked project experiments total 1566 nominal GPU-seconds (.435 GPU-hours); this does
not include the user's other work or replace allocation accounting. Both jobs terminal.
No installation or shared-checkout update. Full logs, cancelled attempt, submitted
recipe and accounting are under experiments/block_sparse_cuda/gh200_qualification/
Arrhenius_GH200/real-nca-01. CPU/GPU raw files live in their respective benchmark folders.

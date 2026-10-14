# Matched CPU and pointer-table scalar batches

Code `2091ba8`. All 17 isolated cases / 408 samples passed independent per-lane CPU
screening and exact replay. Raw [samples.csv](samples.csv), paired
[summary.csv](summary.csv) and [manifest.json](manifest.json) retain every route,
source/binary/header fingerprints and command. GPU-only scalar-01 is preserved.

CPU packed controls use cached LU, hoisted shared rank transforms and elementwise
convolution on the same inputs, charging CPU packing/scattering. CPU intermediate
allocation is included. GPU pointer tables are prepared/uploaded during setup;
timing includes kernel gather/scatter and synchronization. Original per-lane copy
routes remain as controls. All comparisons are single-thread CPU and fixed inputs,
without descriptor refresh/setup or host-device transfer in the measured segment.

| Lambda | Rank | Lanes | CPU independent ms | CPU packed ms | GPU copies ms | GPU kernels ms | CPU packed / GPU kernels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 23 | 1 | 0.0151 | 0.0071 | 0.2790 | 0.2764 | 0.026 |
| 40 | 23 | 2 | 0.0303 | 0.0081 | 0.3390 | 0.3266 | 0.025 |
| 40 | 23 | 4 | 0.0611 | 0.0113 | 0.3743 | 0.3422 | 0.033 |
| 40 | 23 | 8 | 0.1217 | 0.0172 | 0.5329 | 0.4660 | 0.037 |
| 40 | 23 | 17 | 0.2563 | 0.0311 | 0.6078 | 0.4672 | 0.066 |
| 40 | 23 | 32 | 0.4973 | 0.0537 | 0.7375 | 0.4687 | 0.115 |
| 40 | 23 | 64 | 0.9673 | 0.1041 | 0.9840 | 0.4683 | 0.222 |
| 40 | 23 | 127 | 1.9330 | 0.2020 | 1.5533 | 0.4710 | 0.429 |
| 40 | 23 | 256 | 3.8985 | 0.4159 | 2.6012 | 0.4863 | 0.855 |
| 40 | 23 | 512 | 7.7753 | 0.8519 | 4.8590 | 0.4893 | 1.741 |
| 40 | 23 | 1024 | 15.6619 | 1.6659 | 9.5838 | 0.7634 | 2.182 |
| 10 | 15 | 1 | 0.0100 | 0.0044 | 0.2640 | 0.2592 | 0.017 |
| 10 | 15 | 17 | 0.1607 | 0.0187 | 0.4809 | 0.3321 | 0.056 |
| 10 | 15 | 127 | 1.2178 | 0.1225 | 1.4174 | 0.3384 | 0.362 |
| 120 | 32 | 1 | 0.0220 | 0.0106 | 0.2895 | 0.2863 | 0.037 |
| 120 | 32 | 17 | 0.3736 | 0.0467 | 0.6617 | 0.5180 | 0.090 |
| 120 | 32 | 127 | 2.7305 | 0.2974 | 1.6039 | 0.5373 | 0.553 |

Matched CPU materially changes the interpretation of GPU-only batching ratios:
CPU wins small cohorts even against pointer-table packing. Larger cohorts at rank
23 favor GPU, while mesh/rank also matters. This is not a universal lane threshold
or target dispatch rule. GH200 remains the performance target; target timing and a
real ready-work scheduler must establish practical thresholds and memory policy.

Maximum per-lane relative error, including zero/weak lanes, is retained in summary.
CPU work-byte columns are NA (not a measured host peak); GPU explicit work bytes
exclude common buffers and opaque allocations. Independent and packed arithmetic
are compared, not reassociated path accumulation. Full-path cutoffs, irregular
special vertices/correlators, descriptor refresh and changing buffer ownership
remain scheduler requirements. See [methodology](../../README.md).

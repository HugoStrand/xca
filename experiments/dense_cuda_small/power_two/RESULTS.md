# Initial power-of-two screen

2026-10-12, local RTX A4000, CUDA 13.4. These are M1 screening results, not a
production dispatch qualification or GH200 performance prediction. Raw manifests,
failed samples and validation logs are retained in results/.

Six inherited CTest entries pass in mode 2. Expanded primitive tests subsequently
pass dimensions 2/4/8/16/32/64, lanes 1/8/128, both pointer-table vertex directions,
complex alpha and beta=0/1. Adapter tests only replace ordinary SE. Passing primitive
checks does not establish high-order evaluator parity.

Full-API screens use rank 16, capacities 128 or 1, orders 1/3/6/8, crossing/chain,
requested counts 8/128, three timing samples, and individual/aggregate/cache-change
checks. Requested counts truncate to available indices at low order.

| Dimension | Custom products, B128 | Library products, B128 | Serial library, B1 |
|---|---|---|---|
| 2 | Fails strict gate, relative error 1.31e-8 | Same failure | Passes 28 backend/case groups |
| 4 | Passes 28 backend/case groups | Retained E3/E4 controls | Retained E4 controls |
| 8 | Fails strict gate, relative error 3.61e-10 | Passes 28 backend/case groups | Not rerun |
| 16 | Fails strict gate, relative error 1.45e-10 | Fails baseline CUDA/CPU individual check | Same baseline CUDA/CPU failure |

The labelled dimension-16 serial diagnostic isolates an existing order-3 baseline
CUDA/CPU disagreement: norm 6.653361e-8, absolute error 1.658543e-17, relative error
2.492789e-10. Do not attribute that failure to batching or weaken the low-order gate.
The custom candidate also requires separate qualification. Dimension-2 B1 passing
while B128 fails motivates rank-transform/solve width controls. Dimension-8 library
passing while custom fails rejects immediate promotion of that product specialization.

N4 custom B128 full API: crossing orders6/8 6.827/9.170 ms; chain 6.753/8.955 ms.
N8 library B128: crossing 45.539/61.842 ms; chain 45.355/61.508 ms. These are single
process screens, not the three-process repeatability gate for new dispatch decisions.

A single primitive screen (rank16, B128, device events, 100 repeats after ten warmups)
gives the following product times. It excludes evaluator/setup/transfer costs.

| Dimension | Custom ms | Library ms |
|---|---:|---:|
| 2 | 0.00373 | 0.59169 |
| 4 | 0.00965 | 0.59123 |
| 8 | 0.04632 | 0.58314 |
| 16 | 0.29167 | 0.60008 |
| 32 | 2.06279 | 2.17073 |
| 64 | 14.97961 | 14.20004 |

This simple kernel reaches primitive parity around32–64 on this device, but numeric
failures preclude using that as the evaluator dispatch cutoff. Numerical diagnosis,
multiple processes/seeds/ranks, bounded high-order coverage, larger physical fixtures,
resource/profiling controls and target-hardware qualification remain outstanding.
Keep native/library alternatives and the whole baseline fallback available while
building full evaluator functionality. Preserve the failed candidates for later
block-sparse work; do not silently promote them.

## Arithmetic controls and surviving candidates

The tile4 build groups each real-dot stage in four-term chunks. On sampled primitive
inputs it agrees exactly with cuBLAS for all tested dimensions2–64. A current-source
N8 full-API screen passes28 groups; crossing order6/8 B12820.169/27.724ms versus
library45.539/61.842ms. Three-process N8 repeats and wider qualification remain pending.
The earlier preliminary tile4 run is explicitly excluded from qualification in its
manifest because the binary predates finalized diagnostic source; the current run
and primitive source/binary hashes supersede it.

N2 diagnostics: per-lane rank GEMMs alone fail. Per-lane LU solves, with wide rank
GEMMs retained, pass. Groups of2/4/8/16 RHS lanes fail the strict evaluator gate;
non-divisible group3 DLR coefficient checks also fail (negative controls retained).
Group1 plus per-lane rank GEMMs passes extended N2/N4/N8 DLR tests, including tails
and delayed uploads. Three independent N2 per-lane-solve processes pass8 high-order
backend/case groups each, but full batches cost approximately95/131ms at orders6/8,
slower than matching CPU. This is an arithmetic-safe candidate with substantial
launch overhead, not a performance recommendation. No tolerance was changed.

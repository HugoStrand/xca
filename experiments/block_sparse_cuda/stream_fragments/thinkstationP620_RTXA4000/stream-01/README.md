# Streamed correlator fragments: RTX A4000

Source90e04ac, commands and binary hashes in manifest.json. Seven cases/three routes/
three warm samples each:63 rows. Route order reversed on alternating cases, one OMP/
BLAS thread, rank23, one selected OCA topology/index. Reference and host-input retain
all correlator fragments; stream uses one slot per side. All process statuses zero,
finite/CPU-screen/exact self-replay controls pass, signal/error/structural fields agree
offline across all three routes. Transfers, recomputation, allocations and drains included.

| Case | Reference ms | Host input ms | Stream ms |
| --- | ---: | ---: | ---: |
| n4-number-spgf | 56.765184 | 56.469700 | 73.280782 |
| n6-number-spgf | 189.890276 | 190.344718 | 243.339540 |
| n8-number-spgf | 585.810609 | 620.290663 | 858.090792 |
| n6-mixed-spgf | 178.956076 | 174.616216 | 260.928010 |
| n4-number-dynamic-correlator | 18.348276 | 18.416093 | 20.613564 |
| n6-number-dynamic-correlator | 32.788350 | 31.786682 | 39.004180 |
| n10-number-oca | 1106.468692 | 1116.822032 | 1111.565222 |

N8 SPGF bound69493792/55741632/48072512 bytes (reference/host input/stream), sampled
global deltas27262976/20971520/16777216 bytes. Stream latency46.5% above reference,
38.3% above host input: a capacity tradeoff, not a default improvement. Mixed N6 SPGF
stream260.93 vs host input174.62 ms (49.4% overhead). N10 SE unchanged algorithm,
about1112 ms and265175200-byte bound for stream, versus baseline1106 ms/450012768.
Three samples/one index do not establish full-sum/crossover/target performance.
Global free-memory deltas are sampled, not continuous/process-local peaks; small
cases can show zero due to caching/granularity.

Twelve native/library controls pass10.60/10.72 s; library native flags both OFF. CPU
configure omits experiment. Focused four-control memcheck33.378 s, zero errors/leaks.
Inherited controls cover dynamic/high-order/correlator/tiny-root APIs. Added exact
baseline, changed input/mask/output placement, tight budget/rejection/recovery and
repeated endpoint-cache controls. Duplicate endpoint lists prove cache reuse; calls/
indices/changed G rebuild, empty and self-energy entries reset observer. Eighteen SE
adjacent-cutoff probes remain exact; these protect the unchanged SE route, not a new
correlator output cutoff (there is none). Correlator exactness/tiny-root coverage is
provided by the C++ controls. No forced ordered/mapped, broad-suite, Python/MPI or
target-hardware qualification is claimed.

Metadata-only formula: T=16*r*D^2, F=16*r*sum(binomial(N,q)^2). New correlator base
8T+fixed versus host input2F+7T+fixed; endpoint/result/ones extras additional. Fixed
includes five rank tables, pivots/status, two r*2 kernels,32 MiB sparse reserve and
one rank trace. One opposite slot can overlap the seven-tensor fragment phase;
old slot is drained and freed before recomputing. Endpoint phase needs only five
rank tensors. No host fragment array/copy exists. Caller host input and result, model/
atom construction, opaque allocations and headroom remain additional.

N14/r64 stream base89.895 GiB versus host input155.179 GiB; N14/r26 stream36.538 GiB.
Neither is physical rank/accuracy or complete target fit proof. Observed GH200 total
95.58 GiB leaves limited margin; actual free memory/opaque workspaces/host peaks must
be checked in a bounded phase/map probe before a full physical N14 workload. RealBatch
arena remains explicitly unsupported by the host-input experiment. Keep resident
reference for latency, stream as private capacity finalist; qualify policy combinations
and target behavior before C++/Python/MPI integration. GH200-class target; A4000 local
screen. No Arrhenius GPU allocation used, no production/default change.

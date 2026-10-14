# Compact DLR composed edges

Implementation `13ab3b0`. Six isolated cases / 36 rows compare production cached
DLR with the private compact snapshot. Each timed edge includes two LU solves,
packing/status synchronization, convolution, allocation/release and completion.
Inputs/caller buffers/tables are resident; result download/check is outside timing.
All warm and repeated outputs agree exactly in this finite suite. Three samples
per route in one process are not independent process repetitions.

Four primitive controls and eight complete-evaluator controls passed. Primitive
memcheck/full leak checking reported zero errors and zero leaked bytes. Verification
was terminal before timing. Sources/executable hashes and commands are in the
[manifest](manifest.json); medians/storage are in [summary](summary.csv) with raw CSV.

| M / K / N / lambda | Cached ms | Compact ms |
| --- | ---: | ---: |
| 1 / 1 / 1 / 20 | 0.3006 | 0.3076 |
| 16 / 16 / 16 / 20 | 0.5096 | 0.5031 |
| 256 / 256 / 256 / 20 | 51.0904 | 51.3459 |
| 3 / 5 / 2 / 120 | 0.6050 | 0.5939 |
| 3 / 5 / 2 / 20 | 0.5040 | 0.5016 |
| 64 / 96 / 48 / 20 | 3.0177 | 3.0347 |

Observed latency differences are small in this screen; no 15% speedup is claimed.
For the 256-square case (rank18), cached retains 113,272,204 bytes and compact
75,523,468 bytes after convolution, saving 36 MiB. The subsequent solve temporarily
retains an RHS with existing scratch, bringing compact boundary peak to 94,397,836
bytes: peak reduction is 18 MiB, not 36. These values exclude caller buffers, opaque
allocations and host storage; method-boundary observations are not continuous peaks.

Aliased outputs retain an owned diagonal and can need the original five convolution
buffers. Tests cover switching alias modes, exact production comparison, budget
rejection before writing and recovery. Production DLR/ABI/defaults are unchanged.
GH200 remains the target. See [generic methodology](../../README.md) and the
[complete-call comparison](../../../../../benchmarks/block_sparse_common/thinkstationP620_RTXA4000/compact-full-01/README.md).

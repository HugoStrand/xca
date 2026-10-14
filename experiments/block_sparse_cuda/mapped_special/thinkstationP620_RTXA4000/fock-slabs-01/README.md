# Actual Fock-sector mu row slices

Implementation `1cc80db`. Eight isolated processes / 96 samples qualify one mu
row slice using all canonical fields in adjacent number sectors, in both directions.
Three repetitions per route remain one process, not independent trials.
[Manifest](manifest.json) records sources, executable hash, commands and hardware;
[summary](summary.csv) and raw CSV files retain timings and numerical errors.

Fourteen-mode dimensions are 3003x3432 (annihilation) or 3432x3003 (creation), with
1716 entries per field. Known four-mode signs and fourteen-mode occupancy/complement
counts pass independent CPU-only tests. CPU and CUDA host suites pass three cases.
The creation slice (rank3, rows17) passes memcheck/full leaks with zero errors and
zero leaked bytes; its timings are excluded. Verification was terminal before
this unprofiled campaign. No Hamiltonian/atom_diag or fourteen-mode full evaluation.

| Modes / rank / rows / direction | Serial CPU ms | Separate GPU resident ms | Mapped GPU resident ms | Maps + download ms |
| --- | ---: | ---: | ---: | ---: |
| 10 / 26 / 32 / annihilation | 7.460 | 0.776 | 0.297 | 0.825 |
| 10 / 26 / 32 / creation | 6.579 | 0.657 | 0.251 | 0.738 |
| 14 / 26 / 128 / annihilation | 682.044 | 43.813 | 20.281 | 111.836 |
| 14 / 26 / 128 / creation | 637.464 | 38.943 | 17.784 | 96.341 |
| 14 / 26 / 32 / annihilation | 170.772 | 11.308 | 5.123 | 29.299 |
| 14 / 26 / 32 / creation | 161.608 | 10.081 | 4.497 | 24.698 |
| 4 / 3 / 3 / annihilation | 0.001 | 0.106 | 0.011 | 0.051 |
| 4 / 3 / 3 / creation | 0.001 | 0.105 | 0.010 | 0.047 |

Matched CPU uses the same ordered mapped products/weights/additions, allocation and
host output, compiled with one thread. It is a serial control, not a tuned or
threaded CPU ceiling. The separate GPU route retains all operator snapshots and
rank weights outside timing, favoring the reference relative to the production
loop's snapshot rebuilding. Resident timers exclude result download. The final
route rebuilds/uploads maps and weights and downloads mu; its input/output tensors
remain allocated. This is not the complete special vertex or solver pipeline.

At rank26/128 rows, fourteen-mode mapped resident latency is 20.281 ms annihilation
and 17.784 ms creation versus separate 43.813/38.943 ms. Including fresh maps and
host result materialization raises it to 111.836/96.341 ms. Small four-mode CPU
remains faster. Preserve mu on device through its consumers; host materialization
can dominate the benefit. No whole Solver or target hardware speedup follows.

The benchmark caps explicit device tensor/map/operator payload (including fresh
metadata for the inclusive route) at one GiB plus free-memory headroom. CSV reports
tensor/map payload and resident operators separately; opaque handles/allocator and
host arrays are not complete peak measurements. Row slices omit the remaining rows,
left application, sum over mu, path cutoff, DLR work, full output and MPI. They
prove actual-sector mapping and bounded primitive feasibility, not fourteen-mode
full-call admission. That still needs scheduling/streaming and target validation.
GH200-class hardware remains the target, not RTX A4000.

See [methodology](../../README.md). Existing full-call, sparse and cache controls remain.

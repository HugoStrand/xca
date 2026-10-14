# Resident weight selection: complete calls

18 inherited controls passed46.29s; new selector test passed.272s with zero numerical
difference from expanded GPU weights. Targeted selector/multi-set/adjacent-cutoff
memcheck passed three tests17.240s, zero errors/leaks. Four workloads/three routes/
three samples produced36 rows passing finite output, CPU screening and exact replay.
GPU signal/error/count/range summaries match baseline/candidate; no cross-route
bitwise claim is made. Largest absolute error1.570e-16. Rank23, explicit multi-set
fixture, one crossing topology, bounded indices; setup/warm-up separate.

| Case | Matched CPU ms | Parity GPU ms | Resident GPU ms | GPU speedup | CPU/resident |
| --- | ---: | ---: | ---: | ---: | ---: |
| N4 NCA, two indices | 1.183 | 2.233 | 1.792 | 1.25 | 0.66 |
| N4 OCA, eight indices | 4.095 | 4.050 | 3.350 | 1.21 | 1.22 |
| N6 OCA, eight indices | 13.809 | 11.608 | 8.775 | 1.32 | 1.57 |
| N8 OCA, one index | 6.011 | 7.437 | 5.825 | 1.28 | 1.03 |

GPU benefits from reduced host node preparation and transferred payload. Small NCA
still favors CPU; N8 OCA's3% advantage is not robust evidence from three samples.
No whole Solver/topology sum, GH200 crossover or production dispatch claim follows.
Timing ran sequentially after tests/memcheck/builds, before profiling. Resident table
construction is charged to setup. Descriptors/transfers are included in call latency.
Added explicit storage is48,352/71,904/101,344 bytes for N4/N6/N8: persistent tables
plus bounded descriptors, charged before upload and in all observable table bounds.
The inherited cutoff replay band remains experimentally bounded, not universal.

Raw source/binary/runner hashes, commands, samples/summary and qualification logs
are retained. All controls were rebuilt. A4000 is a development proxy; GH200 is target.
Scoped trace lives in the resident experiment's hardware folder and is mechanism
evidence only. No new Arrhenius allocation was used. Next reduce allocation/event/
synchronization frequency with owned reusable staging, then target qualification.

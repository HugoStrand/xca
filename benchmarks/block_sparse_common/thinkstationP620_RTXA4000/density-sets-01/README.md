# Explicit multi-field scalar density-set calls

Four workloads, three routes and three samples produced 36 rows. Every fixture
contains scalar blocks and a grouped multi-field density set. Retarded coefficients
are complex Hermitian. All numerical screens and exact repeated-call checks passed;
CUDA structural counts match, largest absolute error 1.570e-16. Signals are nonzero.
All timing processes ran sequentially after builds, without tests/profilers competing.

Median synchronous API times (ms), rank 23, one crossing topology:

| Workload | CPU | General ordered CUDA | Batched CUDA | CPU/batched | Accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=4 NCA, two indices | 1.140 | 58.444 | 2.823 | 0.40 | 96/160 |
| N=4 OCA, eight indices | 58.976 | 378.847 | 6.302 | 9.36 | 244/640 |
| N=6 OCA, eight indices | 219.656 | 1951.371 | 21.771 | 10.09 | 1328/3584 |
| N=8 OCA, one index | 127.979 | 838.929 | 12.584 | 10.17 | 576/2304 |

The candidate substantially reduces full-call GPU overhead in these multi-field
cohorts; small NCA still favors CPU. CPU is unchanged, not algorithm-matched batched
multi-set arithmetic. These ratios therefore do not isolate GPU hardware gains.
One local campaign/three samples does not establish a robust crossover, complete
Solver/topology-sum speedup or GH200 policy. A4000 is a development proxy.
Setup/warm-up/CPU screening are separate from call latency. CPU setup additionally
exports metadata to validate the fixture. Sampled memory counters retain the common
runner's limitations; they are not continuous peaks or private allocator accounting.

The CPU-only frontend built and its N=4 OCA signal agrees with the CUDA-environment
CPU control. Default scalar CPU fixture still screens exactly (error zero). Wrong
partition selection is rejected. These single samples are compatibility evidence.
Commands, binary/runner/source hashes and raw rows are retained; manifest's compiled
parent revision plus modified-source hashes pins the measurement, while
qualification.json records the subsequent code commit and build/hardware settings.
Next implement an algorithm-matched multi-set CPU control and qualify a bounded
GH200 cohort before promotion. No target allocation was used here.

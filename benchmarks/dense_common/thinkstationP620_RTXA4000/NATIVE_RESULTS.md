# Native workload repeats


RTX A4000, CUDA 13.4; local control for the GH200-class target. Source revisions and runtime details are recorded in the dataset manifests. Each dimension has three fresh processes, seeds 31415/1729/31415, three timed samples per case, and a separately reported first call. The unchanged strict CUDA gate checks individual indices and aggregates; high-order CPU differences are diagnostics. Crossing topology, rank16, capacity128, two poles, ordinary and paired self-energy, orders6/8, index counts128/129.

| N | Method | Order | 128-index median range (ms) | CPU speedup range | 129-index median range (ms) |
|---|---|---|---|---|---|
| 4 | se | 6 | 6.875–6.941 | 13.76–14.00× | 9.370–9.409 |
| 4 | se | 8 | 9.181–9.214 | 14.55–14.62× | 12.638–12.651 |
| 4 | pairs | 6 | 6.926–6.972 | 14.02–14.04× | 9.480–9.522 |
| 4 | pairs | 8 | 9.259–9.284 | 14.42–14.70× | 12.775–12.801 |
| 8 | se | 6 | 20.172–20.235 | 12.72–12.95× | 22.827–22.926 |
| 8 | se | 8 | 27.714–27.734 | 12.84–13.04× | 31.291–31.347 |
| 8 | pairs | 6 | 20.599–20.611 | 12.77–12.97× | 23.323–23.348 |
| 8 | pairs | 8 | 28.135–28.253 | 12.79–13.01× | 31.840–31.860 |

Ranges include process and seed variation; they are not confidence intervals. Raw samples, matched index lists, finite-error metrics, first-call/setup/context costs and workspace reports are retained in the six `native-lifetime-r16-n*-*` directories. All comparisons passed. Each run manifest includes source and binary provenance.

A timed-region Nsight capture (`native-profile-before-reuse/`) records cold and warm calls for N4/order6/128 ordinary indices. Warm work has396 kernels, about5.92ms summed kernel time under profiling, one4096-byte H2D upload and one4096-byte D2H result download. The capture also records one4096-byte unified-memory migration in each direction (solver status traffic); remaining copies are device-to-device. Grouped product kernels use40 registers/thread and no local-memory spill in this capture; library rank/LU kernels dominate time. Profiling overhead means these durations are not interchangeable with unprofiled benchmark medians.

The [warm allocation-free capture](../../dense_cuda_small/thinkstationP620_RTXA4000/results/native-profile-after-reuse/)
records no CUDA allocation/free APIs, one4096-byte H2D upload, one4096-byte D2H
output copy and one unified status migration in each direction. It records267
device-to-device copies versus268 in the other capture. The two captures use the
source versions identified by their manifests. These results establish residency
only for this profiled ordinary case, not every method/shape/cache transition.
Full crossover, dynamic-channel throughput and GH200 measurements need separate qualification.

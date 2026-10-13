# Native workload repeats

Evidence after reorganization: [CPU/shared baseline](../../dense_cpu/thinkstationP620_RTXA4000/baseline/), [native and larger-dimension runs](../../dense_cuda_small/thinkstationP620_RTXA4000/results/), [Solver workload](solver-workload-current/). Paths in historical commands below refer to the recorded revision.

RTX A4000, CUDA 13.4; local control for the GH200-class target. These records precede the upload/download storage-reuse change. Each dimension has three fresh processes, seeds 31415/1729/31415, three timed samples per case, and a separately reported first call. The unchanged strict CUDA gate checks individual indices and aggregates; high-order CPU differences are diagnostics. Crossing topology, rank16, capacity128, two poles, ordinary and paired self-energy, orders6/8, index counts128/129.

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

Ranges include process and seed variation; they are not confidence intervals. Raw samples, matched index lists, finite-error metrics, first-call/setup/context costs and workspace reports are retained in the six `native-lifetime-r16-n*-*` directories. All comparisons passed. Each run manifest transparently backfills the previously omitted entry-source hash from its recorded Git revision. The recorder now includes that file automatically.

A timed-region Nsight capture (`native-profile-before-reuse/`) records cold and warm calls for N4/order6/128 ordinary indices. Warm work has396 kernels, about5.92ms summed kernel time under profiling, one4096-byte H2D upload and one4096-byte D2H result download. The capture also records one4096-byte unified-memory migration in each direction (solver status traffic); remaining copies are device-to-device. Grouped product kernels use40 registers/thread and no local-memory spill in this capture; library rank/LU kernels dominate time. Profiling overhead means these durations are not interchangeable with unprofiled benchmark medians.

Warm CUDA runtime records also show two4096-byte allocations/frees. Source inspection identifies NDA regular-array upload assignment and by-value device-array conversion during result download; a view-based reuse change is under validation. No allocation-free claim is made for these pre-change records. First-call capture includes batch/cache construction and many allocations.

Full dimension/rank/method crossover, dynamic-channel performance, explicit host/device memory accounting and GH200 measurements remain pending; these measurements qualify only the stated local ordinary/paired cases.


After assigning uploads and downloads through views, the matching cold/warm capture
in `native-profile-after-reuse/` has no CUDA allocation/free APIs in its warm call.
It keeps one4096-byte H2D upload and one4096-byte D2H output download; the two unified
status migrations remain. Device-to-device copies decrease from268 to267, removing
the temporary result copy. This proves residency only for this profiled ordinary
case, not every method/shape/cache transition. Strict native/Python tests pass.

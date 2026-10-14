# Scoped multi-field scalar call trace

One warmed N=6 OCA range, first1/count8, explicit shifted-density fixture, rank23.
Numerical CPU screening and exact replay passed. Setup and CPU control were outside
capture; Nsight2026.3.2 traced CUDA only. The wrapper's ~1.73s includes export effects
and is not benchmark evidence. Unprofiled timings remain in density-matched-01.

Runtime trace spans25.890ms, summed API intervals9.134ms. GPU kernel intervals total
2.452ms (255 events); memory operations total about.682ms. Interval sums/categories
can overlap and tracing affects latency; do not derive a promised speedup by subtraction.

| API/event | Calls |
| --- | ---: |
| cudaStreamSynchronize | 272 |
| cudaMalloc / cudaFree | 156 / 158 |
| cudaMemcpy / cudaMemcpyAsync / cudaMemcpy2D | 117 / 57 / 6 |
| cudaEventRecord / cudaStreamWaitEvent | 258 / 258 |
| cudaEventCreateWithFlags / cudaEventDestroy | 276 / 276 |
| Host-to-device copy events | 115 |

TRSM/GEMM dominate GPU arithmetic. Small copies, allocation and synchronization
remain frequent. Source inspection identifies repeated special-field assignment
and parity calculation inside the DLR-node weight loop, although both are constant
per lane/operator pair. Hoisting that work is the next isolated candidate; the
setter changes orbital indices only. Broader resident weight tables/device packing
could then remove per-operation uploads, subject to ownership/admission and numerical
qualification. No production or GH200 inference follows from this local trace.

Text stats and executable/report/source hashes are retained. Binary report and
SQLite database remain under the recorded /tmp paths and may disappear. Capture
ended before any subsequent build/tests. No target allocation was added.

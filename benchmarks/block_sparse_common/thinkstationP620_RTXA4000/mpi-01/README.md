# ThinkStation MPI controls

Benchmark0a55b82, Threadripper PRO3945WX/RTX A4000 development proxy. No target
threshold is inferred. Shared method is in [the generic README](../../README.md).
CPU number two-rank and mixed-retarded four-rank/count1 controls pass. The latter
retains empty owners and tails. CUDA ordered/mapped number two-rank/three-sample
control and library mixed-retarded two-rank controls pass. Selected outputs agree
with a rank-zero CPU indexed reference; indexed ownership variants are exactly equal.

CPU/CUDA data retain command lines, individual jobs, per-rank phases, summaries and
source hashes. Early CPU/library pilot manifests predate the metadata-only extension
hash addition; their actual source hashes remain. Final ordered number records match
0a55b82 and include loaded extension hashes. They are not one shared fresh-process
performance cohort. Number CUDA critical medians: topology split84.565ms,
flat cyclic105.096ms, Solver ranges84.770ms. Better cyclic imbalance does not imply
better latency when two contexts share one GPU. No one-GPU-per-rank inference.

Duplicate job ownership is rejected before summary creation (validation-control.json).
A deliberately sleeping CPU-only worker prefix hits the one-second launcher timeout,
returns124 and leaves no launcher/process group; command/status/log retained. This
control does not submit CUDA work. GPU verification runs are separate, never concurrent
on this device. Known local X11 authorization warnings are startup diagnostics, not
tool approval rejection. No runtime code changes or new full-suite claim.

Separate after-warmup Nsight capture uses CUDA profiler API boundaries for all measured
policies, one sample each. Captured PIDs/ranks:554556/rank0 and554553/rank1. Device copy
exports show rank0 H2D788copies/139472bytes/.347ms and D2H180copies/87360bytes/.217ms;
rank1 H2D545copies/102056bytes/.246ms and D2H127copies/67696bytes/.151ms. Exact bytes
are in device-activity.json; CSV MB columns are rounded. These are device activity
sums for the captured cohort, not end-to-end transfer wall time or policy-specific
costs; profiler overhead and context contention preclude latency comparisons.

Device activity JSON stores each operation as count,total_duration_ns,min_start_ns,
max_end_ns[,bytes], plus raw CUPTI copyKind groups. Nsight exports retain operation
names, counts, durations, sizes and kernel summaries. Original .nsys-rep/SQLite hashes
are retained; binary trace files remain temporary and are not required for CSV replay.
Raw source/report hashes precede archive LF/trailing-whitespace normalization.

[CPU records](../../../block_sparse_cpu/thinkstationP620_RTXA4000/mpi-01/README.md),
[CUDA records/profile](../../../block_sparse_cuda/thinkstationP620_RTXA4000/mpi-01/README.md),
and [GH200 comparison](../../Arrhenius_GH200/mpi-01/README.md) retain scoped evidence.
No Arrhenius allocation was used for local controls. Current target accounting lives
in the target report; this experiment supplies no large-system or transfer-bandwidth
qualification on GH200.

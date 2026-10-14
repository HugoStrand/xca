# GH200 MPI ownership and phase measurements

Job3854536 completed0:0 on n535, one GH200/two MPI ranks, in28seconds (extern row29s).
Benchmark0a55b82/recipe2713bb0 reuse the previously qualified Python extensions from
source7b5d223/runtime85ecb2d. All extension hashes match the qualification record.
Arm/GCC14.3/CUDA12.9, one OpenMP/BLAS thread per rank. No rebuild or shared install change.
Site mpprun explicitly launches srun/PMIx; worker communicator/collective preflight passes.

Two fixed order-three topologies, rank23, bounded windows. Number uses first0/count3,
three warm samples; mixed-retarded uses first0/count1/two samples; occupation-tail
uses first7/count3/two samples. Each case first warms all three policies. These are
samples within one two-rank process group per case, not fresh-process repetition.

Median of per-sample critical-rank wall times (ms), followed by local API-time
imbalance (max/mean across ranks):

| Case | Topology split ms / imbalance | Flat cyclic ms / imbalance | Solver ranges ms / imbalance |
| --- | ---: | ---: | ---: |
| number | 113.834 / 1.052 | 166.407 / 1.003 | 116.192 / 1.047 |
| mixed-retarded | 0.776 / 2.000 | 4.727 / 1.034 | 1.140 / 1.720 |
| occupation-tail | 21.459 / 1.322 | 58.274 / 1.037 | 19.361 / 1.298 |

Cyclic ownership balances work better but is slower in all selected windows. No
scheduler change is justified by these measurements. Multiple contexts share one
GPU; these timings do not model one GPU per rank or multiple GPUs/nodes. The number
window retains all six nonzero indexed jobs. The mixed-retarded selected window is
entirely zero output, and occupation-tail retains both zero and nonzero jobs (16of24
recorded job rows are zero across policies/samples). No active-only filtering.

Moving indexed jobs preserves their output slots **exactly** across both ownership
policies and samples. The Solver-style range grouping passes the CPU screen;
maximum error9.759e-19 across the three cases. Each case uses a rank-zero CPU indexed
control outside measurements. These are selected topology sums, not complete order
or iterative Solver performance/physical accuracy claims.

For number symmetry, one coalesced canonical reduction sends44160bytes per rank;
Solver ranges send14720bytes over eight block collectives. This payload tradeoff is
explicit, not a wire-traffic measurement. Arrival-barrier plus error-coordination
waits are retained separately from payload time. Number Solver-range reduction's
median maximum-rank cost is .060ms, while synchronous API time is116.000ms. The API
includes host/device transfers; it is not isolated kernel time. See the
[generic method](../../README.md) for instrumentation/grouping differences.

Raw case phases/jobs, source/extension hashes and rank metadata are in
[CUDA results](../../../block_sparse_cuda/Arrhenius_GH200/mpi-01/README.md).
[Qualification](../../../../experiments/block_sparse_cuda/gh200_qualification/Arrhenius_GH200/mpi-01/README.md)
retains command lines, module/GPU/accounting/binary checks and source/recipe snapshots.
Local CPU/CUDA/empty-owner controls and separate Nsight transfer measurements are in
[the workstation report](../../thinkstationP620_RTXA4000/mpi-01/README.md).

This closes the bounded MPI phase/ownership measurement gap, not broad scheduling
optimization. Target is GH200-class; no automatic dispatch or production defaults
change. Parent allocation28GPU-seconds brings tracked work to3384seconds/.940000GPUh
against the shared100GPUh/month ceiling. Extern overlap is not another GPU allocation;
site billing and other user work remain separate. All job/steps terminal.

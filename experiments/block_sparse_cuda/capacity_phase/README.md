# Bounded full-shape convolution capacity probe

Target GH200-class accelerators. Local RTX A4000 screens reduced dimensions; full
fourteen-mode Solver/atom construction is deliberately outside this probe.

`block_sparse_capacity_phase` runs the unchanged production DLR convolution on full
rectangular coefficient tensors. It retains a maximum-square opposite fragment and
32 MiB simulated operator reserve, reproducing the expensive streamed-fragment phase
without allocating all sector inputs/outputs or diagonalizing a physical atom. Coefficient
uploads use one host rank plane at a time. Record setup/upload, independent oracle and
first/warm convolution times separately. This is a phase probe, not a complete-call
benchmark or a claim of full physical accuracy.

An independent long-double contraction computes three spatial output entries at every
DLR node, summing the full inner and rank dimensions. It rebuilds the cached double
DLR constants independently and promotes them before its sums/products. Acceptance is
sampled error <=1e-11*sampled oracle norm +1e-24. Two repetitions additionally require
exact sampled replay. Other output entries are not independently checked. This tests
dense arithmetic at actual dimensions; it does not certify cutoff decisions or the
full physical model. Completed scratch release must leave only immutable rank tables.

Inputs:

- `--m`, `--k`, `--n`: extents 1..3432. Execution above512 requires `--allow-large`,
  an actual GH200 device and a Slurm job environment.
- `--lambda`:10..10000, rank built from cppdlr at eps1e-10, nonsymmetric. Rank is
  measured, never truncated. Lambda4000 gives rank64 locally.
- `--budget-mib`: explicit device payload ceiling, default1024, maximum98304.
- `--samples`:1 or2; first includes scratch allocation, second retains warm scratch.
- `--dry-run`: host-only rank/memory inventory, no CUDA context/payload allocation.

Explicit bound: immutable tables + operator reserve + opposite square tensor + two
coefficient inputs + output + convolution scratch (one tensor per input, two output-
shaped tensors). All arithmetic uses checked sizes/BLAS limits. Budget rejection occurs
before device allocation, then actual free bytes are checked after table/handle setup
with64 MiB headroom before large arrays. Admission status2 distinguishes a declared/
free-memory rejection from numerical/runtime errors(status1). Preserve rejection records;
never retry with a relaxed cap automatically. Opaque allocations may still fail and are
reported. Host peaks and actual global free-memory deltas must be measured on target.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cbuild-cuda/benchmarks/experiments/block_sparse_cuda/capacity_phase/block_sparse_capacity_phase --m 129 --k 128 --n 127 --lambda 4000 --samples 2 --budget-mib 4096
cbuild-cuda/benchmarks/experiments/block_sparse_cuda/capacity_phase/block_sparse_capacity_phase --m 3432 --k 3003 --n 3432 --lambda 4000 --dry-run
```

The second shape is the N14 total-number central/adjacent sector pair. It is metadata
until admitted inside a capped target allocation. N10 uses252/210/252. Intended target
recipe must pin source/dependencies, stage timestamps before allocation, reuse Arm
builds and test reduced controls before one large phase. Check account/queue and the
shared100 GPU-hour/month ceiling. No Cartesian sweep or full N14 CPU reference.

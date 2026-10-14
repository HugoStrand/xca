# Packed scalar DLR edges

Performance target: GH200-class hardware. This E4/E5 experiment groups independent
scalar edges across RHS columns, retaining every DLR node per lane. Production
rectangular products already batch across rank; this is a different level of work
sharing. Production evaluators and per-path magnitude policies are unchanged.

All routes use the same cached LU/getrs, time-ordered convolution formula and
beta=2. Packed convolution uses shared rank transforms and elementwise diagonal/
cross products, avoiding cross-lane outer products. No custom LU or inverse is used.
CPU controls hoist the Hilbert transforms; CPU packing can improve its arithmetic
as well as GPU packing, so matched CPU comparisons are required.

Eight routes on identical owned input buffers:

- CPU independent: two solves and hoisted convolution per scalar edge.
- CPU packed inclusive/resident: combined RHS solves and rank transforms; inclusive
  gathers input, both scatter output. Intermediate CPU allocation is timed.
- GPU independent: retained DlrOps per edge.
- GPU packed inclusive/resident: gather/scatter with per-lane cudaMemcpy2DAsync.
- GPU packed kernel inclusive/resident: gather/scatter using device pointer tables,
  one kernel per gather/scatter. Native arithmetic disables FMA contraction.

Resident routes exclude input gathering but still scatter. GPU pointer tables are
prepared/uploaded during setup, borrowing addresses from buffers owned throughout
the experiment. Changing allocation/address/ownership requires table refresh in a
future scheduler. A post-timing check scales current device input contents in place,
updates the CPU control, and verifies that stable descriptors gather the changed
values. No timing rows mix original and updated input epochs.

Host timers include stream completion and output scattering, exclude context,
initial input upload/descriptor preparation, allocator setup and output downloads
used for checking. Route scratch is retained; CPU intermediates are allocated inside
its calls. One warm-up precedes raw repetitions. Per-edge time is segment time / lane
count, measuring throughput within a cohort, not one independent Solver latency.

Each output lane is compared independently against CPU (1e-12*lane norm+1e-30),
including exactly zero and 1e-10-scale coefficient pairs. A batch norm must not hide
weak lanes. All repeated outputs must match exactly. Numerical errors are retained.
CPU work bytes are NA rather than an unmeasured host peak. GPU explicit work bytes
include route tables/RHS/scratch, excluding common input/output buffers and opaque
allocations; packed intermediate arrays and pointer tables are additional work.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_scalar_edges --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 cbuild-cuda/benchmarks/experiments/block_sparse_cuda/scalar_edges/block_sparse_scalar_edges \
 --lanes 127 --lambda 40 --samples 3
python3 experiments/block_sparse_cuda/scalar_edges/run.py \
 --binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/scalar_edges/block_sparse_scalar_edges \
 --machine thinkstationP620_RTXA4000 --campaign scalar-cpu-01
```

The CUDA-compiler experiment is omitted in CPU-only builds. Lane count is bounded
1–1024, Lambda 1–120, samples 1–5, nonsymmetric epsilon=1e-10 mesh. The recorder sweeps
1/2/4/8/17/32/64/127/256/512/1024 at Lambda40, plus 1/17/127 at 10 and 120 (17 cases,
eight routes, three samples). It stores source/binary/header hashes, exact commands,
errors and timeouts (120 s), and refuses existing campaigns. System results belong
under hardware subdirectories. Original experiments/results remain available.

No topology/path scheduler or cutoff is implemented here. Promotion requires copied
borrowed metadata, bounded ready-work memory, individual completed-path decisions,
original accumulation order, changing-input/descriptor lifecycle and full-observable
coverage. Special-vertex sums and irregular correlator fragments need explicit
handling; this segment does not establish complete Solver speedup or target dispatch.

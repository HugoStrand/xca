# Packed scalar DLR edges

Performance target: GH200-class hardware. This E4/E5 experiment groups independent
scalar edges across RHS columns while keeping all DLR nodes for each lane. The
production rectangular product already batches across rank; this tests a different
parallelism level. Production evaluators and their per-path magnitude policy are
unchanged. No topology/branch scheduler is implemented here.

The independent route uses retained DlrOps for two cached-LU solves and convolution
per edge. Packed routes use the same LU/getrs on a rank-by-lanes RHS, shared rank
transforms, and ordered complex-double elementwise diagonal/cross products (native
FMA disabled). Each lane remains independent: no lane-by-lane outer product and no
sum across paths. The convolution formula and addition order match the hoisted CPU
control, including beta=2. No custom LU solver or inverse is introduced.

`packed_inclusive` gathers inputs from separately owned device edge buffers with
cudaMemcpy2DAsync, computes the batch and scatters to separate outputs. `packed_resident`
excludes input gathering but still scatters. `independent` naturally consumes and
produces separate buffers. Host timing includes stream completion, excludes initial
upload/context/allocation/CPU control and output downloads used for checking. Rank
scratch is retained. One call warms each route before three samples. Per-edge latency
is total segment time divided by lane count; that is throughput within the batch,
not the latency of one independent Solver evaluation.

Each output lane is compared independently against CPU (1e-12*lane norm+1e-30),
including exactly zero lanes and 1e-10-scale inputs with much weaker outputs. This
avoids a batch-wide norm hiding dropped weak lanes. Repeated outputs must match
exactly. Numerical errors are retained. Explicit work bytes include route tables/
RHS/scratch, excluding common input/output buffers, opaque library allocations and
complete process peak. Packed f/g and intermediate arrays are additional route
workspace. Changing data, errors/pruning and accumulation require separate scheduling
qualification before use in a diagram evaluator.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_scalar_edges --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 cbuild-cuda/benchmarks/experiments/block_sparse_cuda/scalar_edges/block_sparse_scalar_edges \
 --lanes 127 --lambda 40 --samples 3
python3 experiments/block_sparse_cuda/scalar_edges/run.py \
 --binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/scalar_edges/block_sparse_scalar_edges \
 --machine thinkstationP620_RTXA4000 --campaign scalar-01
```

The CUDA-compiler experiment is omitted in CPU-only builds. Lane count is bounded
1–256, Lambda 1–120, samples 1–5, nonsymmetric epsilon=1e-10 mesh. The recorder sweeps
1/2/4/8/17/32/64/127/256 lanes at Lambda=40, plus 1/17/127 at 10 and 120. It retains
source/binary/header hashes, exact commands, errors and timeouts (120 s); existing
campaigns are refused. System results belong under hardware subdirectories.

The extended matched screen adds timed CPU independent/packed routes using the same
cached LU and hoisted transform formula. CPU inclusive packing/scattering is charged;
CPU intermediate allocations are inside timing and CPU work bytes are `NA` (not a
measured host peak). It also adds GPU pointer-table gather/scatter routes: one kernel
per pack/scatter instead of per-lane copies. Tables are uploaded during setup and
borrow addresses from edge buffers that remain owned/alive throughout the run.
If buffers move, a future scheduler must refresh those tables; this is not an
ownership guarantee from the descriptors themselves.

The extended bound is 1–1024 lanes, adding 512/1024 at Lambda40. The new default
campaign is `scalar-cpu-01`, 17 cases and eight routes per case, three samples each.
The original GPU-only `scalar-01` results remain intact. CPU packing also improves
its arithmetic; compare matched CPU rather than only GPU independent/packed ratios.
Pointer-table setup, changed input/allocator costs and full diagram scheduling are
still outside this segment screen and require later qualification.

# Hoisted DLR convolution control

Performance target: GH200-class hardware. This E5 experiment isolates an algorithmic
CPU scheduling change before attributing complete-evaluator gains to CUDA.
Production CPU and CUDA evaluators are unchanged.

The installed cppdlr rank-three time-ordered convolution computes both full-input
Hilbert transforms inside the DLR-node loop. `Hoisted` uses the same double helper
tables, NDA products and addition order, computing each transform once. Both
coefficient-only convolution and a complete edge (two cached-LU coefficient solves
plus convolution) must match the unchanged CPU result bit for bit. CPU and CUDA
replays must also repeat exactly. The CUDA numerical screening gate is
1e-10*CPU norm+1e-14; observed errors are retained, and strict primitive tests remain
separate. Inputs are deterministic complex coefficient tensors with unequal spatial
shapes, not a physical complete diagram. Rank/mesh and solves are shared.

With `Build_Benchmarks=ON`, this isolated target builds in CPU and CUDA trees:

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_dlr_hoisted --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 cbuild-cuda/benchmarks/experiments/block_sparse_cuda/dlr_hoisted/block_sparse_dlr_hoisted \
 --m 56 --k 70 --n 56 --samples 3 --lambda 40
```

Dimensions are limited to 1–252, samples to 1–5 and Lambda to 1–120; epsilon is
1e-10, beta is 2 and the mesh is nonsymmetric. Output is raw CSV. One untimed call
warms each route. CPU timers include temporary allocation; GPU scratch is retained.
The CUDA timers include synchronization, output allocation and device-to-host
transfer, with input already resident. They exclude initial uploads, context/handle
construction and sparse vertices; these are primitive controls, not full-call
comparisons. Reported explicit device bytes include caller inputs/outputs and DLR
scratch/tables, excluding opaque library allocations. No dispatch is promoted here.

System-specific results belong below informative hardware subdirectories. Sweep
scalar, small square, unequal rectangular and bounded large-sector shapes. A
matched full-evaluator CPU control and changed-input timing are still required
before interpreting complete diagram or Solver speedups.

Record the eight-case bounded CPU/CUDA sweep with the CUDA binary:

```bash
source ~/env/xca-cuda.sh
python3 experiments/block_sparse_cuda/dlr_hoisted/run.py \
 --binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/dlr_hoisted/block_sparse_dlr_hoisted \
 --machine thinkstationP620_RTXA4000 --campaign hoisted-01
```

The recorder uses isolated processes with 120 s timeouts and refuses overwriting
campaigns. Raw CSV, exact commands, source/binary/cppdlr-header hashes and failures
are retained under the machine directory. CPU-only builds can run the executable
for qualification but the paired recorder requires all six CPU/CUDA route-stage
combinations. The large 210-by-252-by-210 case uses one repetition; it represents
a bounded ten-mode sector geometry, not a complete ten-mode diagram.

# Bounded scalar self-energy path scheduler

Performance target: GH200-class hardware. This private E4/E5 candidate batches
complete scalar self-energy paths, retaining general evaluator execution for other
block shapes and all correlators. Production classes/bindings are unchanged.

Eligibility requires every propagator block to be 1x1 and each symmetry set to have
one field. This is an initial operation-cohort capability, not a reduced evaluator
API: unsupported cases execute the unchanged general path. Arbitrary scalar sets
with multiple mu/kappa members need separate scheduling qualification.

The generated class reuses current reference code, inserting a scalar worker and
changing the self-energy visitor. Each cohort has at most 512 copied Backbones and
path descriptors; owned NDA arrays/vertices survive reuse of the borrowed visitor.
Cohorts split when special-vertex positions differ and retain discovery order.
Packed cached-LU solves/rank transforms come from the qualified scalar-edge component.
No custom solver/inverse or path reassociation is used.

Vertex application, pole weights, convolution, special kappa/weight/mu operations,
prefactor and sign execute in the original sequence. Each completed lane receives
its own max-absolute cutoff (>1e-16); accepted lanes accumulate in original order
into their output blocks. Unused tail lanes stay zero and are never accumulated.
Input G is gathered through current device buffers, not cached across changed calls.
Weights, descriptors, transfers, host cutoff scan and ordered fan-in are all charged
to complete-call timing. Batch storage is created lazily, avoiding construction for
zero-survival windows. The additional conservative workspace is admitted/reported
before input upload; host metadata/opaque allocations are not complete peak admission.

The private class exposes the same API. Its test executable runs the unchanged eight
CUDA evaluator tests against it. Four independent qualification tests also cover multi-cohort/tail range sums,
changed data/masks, adjacent NCA cutoff values, additional-workspace rejection
before submission, empty ranges and multi-field scalar-set general execution.
Additional multi-cohort/tail benchmarks compare CPU and require exact GPU replay. No Python/MPI selector exposes this candidate.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_scalar_path_tests \
 block_sparse_scalar_path_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild-cuda \
 -R '^experiments/block_sparse_cuda/scalar_paths$' --output-on-failure
python3 benchmarks/block_sparse_common/run_calls.py \
 --cpu-binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/dlr_hoisted/block_sparse_cpu_hoisted_benchmark \
 --cuda-binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/scalar_paths/block_sparse_scalar_path_benchmark \
 --machine thinkstationP620_RTXA4000 --campaign scalar-path-01
```

Generic definitions stay here; shared hardware/campaign results use the benchmark
backend folders, with paired comparisons in block_sparse_common. The retained native,
library, unchanged/matched CPU and edge experiments remain available. Full-candidate
CPU batching, irregular special sets/correlators, higher-order cutoff-adjacent paths, wider shapes,
changed-address ownership and target dispatch remain pending. Selected-index speedups
do not establish complete Solver/topology-sum throughput or N=14 qualification.

## Matched full-call CPU control

`block_sparse_cpu_scalar_path_benchmark` also builds in a CPU-only tree. Its host
scheduler shares recipe.inc with the GPU candidate: scalar eligibility, copied
metadata/cohorts, vertex/edge/special operations, prefactors, cutoff and ordered
fan-in. CPU uses cached LU with packed RHS and hoisted rank transforms. No GPU
arithmetic or native compiler is required for this control. Unsupported shapes and
other observables use the unchanged CPU evaluator (not hoisted/batched controls).

Use this executable as `--cpu-binary` in the paired recorder, campaign
`scalar-matched-01`. CPU rows have no path-count observer/device counters. Shared
recipe extraction leaves the GPU qualification suite unchanged; per-case CPU results
are screened against original CPU and repeat exactly. Compare identical eligible
scalar self-energy rows before attributing any old-baseline speedup to GPU hardware.
Descriptor setup, allocations, host transfers and cutoff/fan-in remain relevant
inclusive costs. Results under the common hardware folder document GPU-losing cases.

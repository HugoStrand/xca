# Scalar symmetry sets with multiple operators

Private candidate, with GH200-class accelerators as the performance target. The
earlier singleton constants/cutoff experiments and production are preserved.

Eligibility requires all propagator blocks to be scalar, allowing symmetry sets
with multiple fields. Special vertices retain the reference's outer μ and inner κ
summation order per lane. Each pair applies κ, its hybridization weight, then the
ordered sum; μ acts after its κ sum. Missing members of shorter sets contribute
zero. Only independent lanes are batched; no atomics or reordered reductions are
used. Singleton cohorts retain their original operation sequence.

Three additional rank-by-512 buffers enter admission. General block shapes and all
correlators retain the general route. This candidate includes the experimental
cutoff replay policy; its fixed band remains unproven for arbitrary conditioning,
orders and cancellation. No production/binding selector or default exposes it.

Build `block_sparse_scalar_sets_tests` / `block_sparse_scalar_sets_benchmark` with
benchmarks/tests and native sparse kernels enabled. CTest is
`experiments/block_sparse_cuda/scalar_sets`. The suite reuses existing scalar/replay
controls and adds mixed singleton/three-field sets with complex retarded coefficients,
NCA/OCA range calls, changed propagator owners/masks, exact replay, empty ranges and
pre-upload workspace rejection. Shifted density operators ensure identical nonzero
block connectivity in this fixture; this does not change production symmetry analysis.

The generic benchmark frontend does not itself guarantee a multi-field set; inspect
the fixture before inferring performance for this capability. Dedicated multi-set
timing and GH200 evidence remain required before promotion.

## Explicit density-set call benchmark

`block_sparse_cpu_density_sets_benchmark`, `block_sparse_general_density_sets_benchmark`
and `block_sparse_scalar_density_sets_benchmark` use a common explicit fixture:
up to three shifted density operators, grouped by identical scalar connectivity,
with complex Hermitian retarded coefficients. Setup validates scalar dimensions
and a multi-field symmetry set; require `--dynamic --partition occupation --observable se`.
CPU uses the unchanged evaluator; general CUDA selects ordered sparse operations;
batched CUDA selects this candidate. The CPU binary also builds without CUDA.

The shared complete-call runner accepts an optional interaction-fixture callback,
leaving its default workload unchanged. Both the backend and its CPU numerical
control receive the same substituted fixture. CPU setup additionally exports a
model for validation; it is charged to setup, not synchronous call latency.
Timing includes discovery, input packing/transfers, special sums, cutoff/replay
and output construction. It excludes setup, warm-up and numerical-control calls.

Use `gh200_qualification/run_cohort.py --cohort density` with a build directory,
output directory and revision JSON to record four bounded N=4/6/8 cases across
three routes, three samples each. Revisions/source and binary hashes are retained.
The unchanged CPU frontend is not an algorithm-matched batched implementation;
use the matched frontend below to separate algorithm and hardware gains. GH200
evidence remains a promotion gate. Ordinary benchmark screens
do not replace the cutoff-adjacent and ownership tests.

`block_sparse_cpu_batched_density_sets_benchmark` adds the algorithm-matched host
control and builds without CUDA. It shares the GPU candidate's lane/cohort discovery,
ordered special-vertex recipe, packed cached-LU solves, transform reuse, cutoff and
fan-in order. Host zero/add operations use scalar loops in place of CUDA operations.
The earlier singleton CPU control retains its eligibility restriction. This matched
control screens against the original CPU and checks repeated outputs; it does not
implement the GPU's cutoff uncertainty-band replay. Compare ordinary full calls
before attributing speedups to GPU hardware. The density recorder now includes this
fourth route; older three-route campaign evidence is retained.

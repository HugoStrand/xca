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

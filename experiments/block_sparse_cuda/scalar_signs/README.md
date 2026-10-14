# Lane-invariant special parity preparation

Private CPU/GPU candidate; preserves scalar_sets and production baselines. GH200
remains the target, with initial local correctness/timing on the RTX A4000.

For each special-vertex mu/kappa pair, field assignment and permutation parity are
computed once per active lane, then reused over DLR nodes. The same integer sign
multiplies the same node-dependent hybridization value in the original order.
Missing set members remain zero; mu/kappa summation order, lane cutoff and fan-in
are unchanged. The bounded host sign vector has at most512 entries per pair. Device
payload and admission are unchanged; no sign cache survives a cohort or call.

`block_sparse_cpu_density_signs_benchmark` and `block_sparse_scalar_density_signs_benchmark`
use the explicit density-set fixture and the hoisted host/GPU recipe. The host
benchmark builds without CUDA. `block_sparse_scalar_signs_tests` reuses all18
multi-set/replay/physics/admission/ownership controls. Older weight callbacks remain
supported by the private adapter; old variants recompute parity as before.

Use `gh200_qualification/run_cohort.py --cohort signs` for four bounded workloads,
four old/new CPU/GPU routes, three samples each. Numerical screening and exact
repeatability precede timing interpretation. The inherited GPU cutoff replay band
remains experimental; host control does not claim that policy equivalence. No
production/Python/MPI selector exposes this candidate. Target qualification and
resident weight/descriptor preparation remain next gates.

# Resident scalar weights and lane descriptors

Private candidate; parity-hoisted and older baselines are retained. Target hardware
is GH200-class, with initial local qualification on the RTX A4000.

Pole kernels reuse the evaluator's existing device vectors. For scalar-eligible
models, constructor-owned forward/reflected hybridization tables are uploaded once
and charged to fixed storage before device allocation. All observable bounds include
that persistent payload, including general correlator fallbacks. A bounded per-lane
descriptor selects a table pointer/stride and integer sign; a device kernel applies
the same weight and complex multiplication sequence. Missing members/identity/zero
descriptors and tail lanes retain their previous behavior. No mutable G is cached.

The descriptor allocation is charged to scalar admission. Default constructor,
context cleanup, coefficient snapshots and changed-input rules remain inherited.
The fixed cutoff replay band is still experimental, not a universal error bound.
GPU selection uses --fmad=false to match existing scalar multiplication arithmetic.

`block_sparse_scalar_resident_tests` reuses18 controls and adds selector equivalence
against expanded GPU weights for strided lookup/sign/identity/zero/update cases.
`block_sparse_scalar_density_resident_benchmark` uses the explicit multi-set fixture;
`run_cohort.py --cohort resident` compares parity-hoisted CPU/GPU and this GPU candidate.
`profile_density_resident` captures one warmed call with the existing profiler-API
workflow; export times are excluded from benchmark claims. Resident construction
appears in setup, while descriptor preparation/transfers are in full-call timings.

This reduces weight payload and host rank-node preparation. Descriptor uploads,
allocations and synchronizations still occur per operation; no claim of their
elimination follows. Next investigate owned reusable staging/descriptor buffers and
fewer synchronization points, preserving host-source lifetimes and stream ordering.
GH200 and broader numerical/integration gates precede promotion. No production or
Python/MPI selector exposes this candidate.

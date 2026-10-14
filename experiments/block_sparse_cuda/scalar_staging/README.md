# Reusable scalar descriptor staging

Private experiment targeting GH200-class GPUs. Older variants remain available;
RTX A4000 measurements are development evidence only.

Each scalar cohort owns reusable host/device buffers for resident-weight descriptors,
constant coefficients and propagator pointers. Copies and kernels use the evaluator
stream. Operation-level completion remains in place before host buffers are rewritten;
the staging destructor also completes outstanding work before destroying storage.
There is no cross-call cache or change to cutoff, path ordering or mutable inputs.

Weight descriptors have nontrivial default initialization. Construct their device
vector from the host vector, then explicitly order and complete that initial upload.
An extent-only NDA device constructor would attempt host initialization of device
storage. Subsequent operations reuse the allocation and upload on the owned stream.

The simultaneously resident constants and pointers add 12,288 bytes at 512 lanes
above the resident-weight variant; this payload is charged to admission. Existing
scalar scratch and resident immutable tables remain charged separately.

`block_sparse_scalar_staging_tests` inherits the resident variant's nineteen controls.
`run_cohort.py --cohort staging` compares the parity-hoisted CPU, resident GPU and
staging GPU on the same four bounded density workloads. Complete-call timings include
staging construction, copies and completion. `profile_density_staging` is available
for scoped mechanism tracing; profiling durations are excluded from timing claims.

This isolates buffer reuse and owned-stream uploads while retaining synchronization.
Further barrier reduction requires separate source-lifetime and stream-order checks.
The experimental cutoff replay band and production/Python/MPI promotion gates remain.

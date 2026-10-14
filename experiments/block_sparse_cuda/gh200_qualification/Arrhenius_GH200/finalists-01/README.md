# GH200 finalist qualification

Job3853285 on n123 completed0:0 in329seconds, one GH200, account
naiss2026-3-569-gpu, fifteen-minute allocation cap. Source d9fa8a2, runtime85ecb2d,
recipe b052919; dependency, runner, binary and staged-source hashes are retained.
Build uses isolated Arm sources/cache and the installed CUDA TRIQS prefix; Python
and dense-small are OFF, native sparse kernels ON. No shared checkout/install change.
Source mtimes were normalized before allocation. Scheduling-only probe3853231
was not an allocation. All GPU work was sequential.

Four focused CTests passed in60.72seconds: asynchronous device fault, mapped special,
real-NCA dispatch and real-NCA numerical controls. The deliberate trap test qualifies
owned-stream error propagation and poisoned evaluator entry; it is not a sanitizer-
clean test or injected mid-topology MPI recovery. This is not a new full suite.

The fixed production cohort completed126fresh processes /378samples, with three
independent processes for every selected case/backend. Full paired results and
limitations are in [the shared report](../../../../../benchmarks/block_sparse_common/Arrhenius_GH200/finalists-01/README.md).

Accounting includes build/test overhead:329GPU-seconds this job. Cumulative tracked
work is3356seconds/.932222nominal GPU-hours, within the shared100GPUh/month ceiling.
This excludes other user activity and does not replace site billing. Job terminal;
no further target allocation was made for this record. Downloaded raw hashes precede
archive text normalization; raw-sha256.json explains file identities.

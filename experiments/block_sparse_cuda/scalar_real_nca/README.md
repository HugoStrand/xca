# Guarded pointwise real scalar NCA

Private prototype over the retained streams scheduler, targeting GH200-class GPUs.
It batches only resident-output NCA with1×1 blocks, exactly real field values,
exactly real forward/reflected hybridization tables and exactly real G entries.
No tolerance treats a small imaginary part as zero. All other inputs/topologies/
observables/output placement retain general CUDA execution. Guards precede batched
submission; runtime GPU failures are not retried through another backend.

NCA contains no DLR solves/convolutions. With finite real intermediates, each scalar
operator/weight application is a single real multiplication, sums retain mu/kappa
and path order, and final signs are exact. The old empirical cutoff replay band is
bypassed in this domain. Range guard bounds every product/partial sum and the whole
possible output sum, reserving half FP64 maximum and limiting sum length. An earlier
product is checked even when a later zero factor would hide its overflow. General
fallback preserves existing behavior outside the guard; it is not new overflow repair.

Immutable model/weight checks are cached; G/range eligibility is checked per call.
No G cache. last_real_nca_lanes reports actual batched path usage; inherited replay
observer remains zero. Tail/changed-input/complex/large-range/high-order/mixed-block/
host-output controls check numeric equality and routing. Signed zero is not a
bit-pattern claim. Subnormal behavior is checked on local hardware and still needs
GH200 qualification.

This inherits deliberately conservative resident tables, arena and unused packed
DLR storage. Admission charges them even for some fallbacks. It is not the final
production memory design. Extract a minimal NCA-only kernel/workspace before exposing
through consolidated bindings/Python orchestration. Higher-order batching/enclosures
remain part of the full goal; NCA/general fallback is an incremental step.

Build block_sparse_scalar_real_nca_tests and block_sparse_scalar_real_nca_benchmark.
The benchmark asserts actual route selection, checks CPU reference/exact replay
through the common runner, and includes fallback cases. Local complete-call results
are hardware-specific; automatic production defaults remain unchanged.

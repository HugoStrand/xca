# Correlator fragment streaming experiment

Target GH200-class accelerators; RTX A4000 runs screen local semantics/mechanisms.
Production, host-input, lifetime and row experiments remain intact.

`StreamFragmentEvaluator` layers two-slot fragment residency on host-backed input and
full-shape lifetime scheduling. Each index owns one right and one left slot keyed by
root. Endpoint traversal/summation order is unchanged. A missing slot is drained and
evicted before recomputing its original full fragment; the other slot remains live.
Slots reset at every index/call, so changed G/backbone cannot reuse old intermediates.
No host fragment cache or all-sector device fragment array is allocated. Tiny starting
roots and the original operator/solve/GEMM/cutoff arithmetic remain unchanged.

During a cache miss, seven phase tensors plus at most one opposite slot require eight
maximum rank tensors. Endpoint application needs two slots plus three products (five).
The bound includes fixed rank tables, sparse/mapped reserves and two result tensors,
trace vector and endpoint-ones vector. Host input and complete host result remain live;
opaque library allocations/free-memory headroom and model construction are additional.
Self-energy follows host-input behavior; explicit RealBatch arena remains unsupported.

Build `block_sparse_stream_fragment_tests`, `block_sparse_stream_fragment_benchmark`
and `block_sparse_stream_fragment_cutoff_audit`. CPU omits the experiment. Tests inherit
production controls plus exact baseline/budget/recovery and repeated endpoint-cache
controls. `last_fragment_builds()` counts successful builds in the latest correlator
call and resets at self-energy/empty correlator entry. It is an experiment observer.

Memory reduction can require substantial recomputation when endpoints cycle through
roots. Measure full-call transfer/allocation/drain time and fragment rebuilds; never infer
speedup/default policy from reduced storage alone. Compare resident-fragment host-input
and unchanged reference on identical fixtures. Hardware-specific results belong in
machine subfolders. No Python/MPI, target or full physical fourteen-mode qualification
is implied by this private C++ experiment.

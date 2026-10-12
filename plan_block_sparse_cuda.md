# Block-sparse CUDA: future-work stub

Date: 2026-10-12. Planning only; implementation follows the operational dense
`c++/triqs_xca/dense/cuda_small` work. Preserve every current experiment and its
negative results. See `plan_dense_cuda_small_generic.md` and
`experiments/dense_cuda_small/COMPARISON.md`.

## Target hardware and orchestration

Performance target: high-end accelerators such as GH200, not the local RTX A4000.
Use A4000 data as a development proxy; measure shape distributions, crossover and
packing/fusion policies on target hardware before choosing production defaults.
Python's API can change: it orchestrates MPI partitioning, rank-to-GPU placement and
collectives. Native code should own bounded work descriptors, local scheduling and
ordered accumulation; do not materialize all rank-local indices in Python or put
collectives inside unequal local chunk loops. Coordinate errors before reductions.

## Problem and intended direction

The atom_diag-based dense solver operates on the full fermionic Hilbert space:
N fermionic modes give dimension d=2^N, so dense-small needs only2/4/8/16/... matrix
sizes. Block-sparse evaluation uses atomic symmetries to decompose that same space
into smaller subspaces. Their individual dimensions need not be powers of two and
vary between blocks, even though the total Hilbert-space dimension remains2^N.
Consequently, the dense power-of-two specialization set is insufficient here:
consider arbitrary small block dimensions and rectangular inter-block products,
alongside library dispatch. Defer that broader kernel coverage to this investigation
rather than making it a prerequisite for the dense-small implementation.

`block_sparse::DiagramEvaluator` follows operator-dependent block paths using
`ind_path` and `block_dims`. Propagator blocks are square, but field operators and
intermediate products can be rectangular. Dimensions and active/zero blocks vary
along a backbone and between indexed evaluations. Choosing one backend using the
maximum block size is unlikely to be sufficient.

Explore operation-level dispatch between small custom kernels and library CUDA
products, coupled to batching across compatible indexed evaluations and DLR nodes.
A dispatch key needs `(M,N,K)`, rank, available lane count, layout/strides, transpose
conventions, accumulation mode and kernel capability. Dense crossover measurements
are a starting point, not a block-sparse dispatch rule. Rectangular shapes, ragged
batches and packing costs require new evidence.

## Topology-level parallelism and shared computations

Also consider the sum over topologies as a parallelization axis, alongside block
paths, indexed evaluations and DLR nodes. `python/triqs_xca/diag.py` generates
pairings recursively and filters connected topologies; the number grows
combinatorially with expansion order. The current Python Solver traverses these
topologies sequentially for SE, paired SE, SPGF and one-time correlators.

Investigate bounded work batches spanning multiple topologies, with scheduling
across MPI ranks and within each GPU. Compare topology, index and combined work
partitioning using measured costs: topology workloads can differ substantially
after block-path pruning. Avoid materializing the complete topology/index product.
Preserve signs, multiplicity and deterministic topology/index accumulation order;
changing the placement or grouping of MPI reductions needs numerical validation.

Look for identical partial computations shared across topologies, such as compatible
backbone prefixes or operator/convolution subpatterns. CUDA graphs may provide a
useful execution representation: compute a common subpattern once and let dependent
branches consume its result, or reuse graph structure across repeated Solver calls.
Graph replay alone does not memoize numerical results; sharing must be represented
explicitly by a dependency DAG and safe buffer ownership. Equivalence must include
operator/pole choices, weights, block paths, layout and propagator version, not merely
similar topology diagrams. Changed inputs must recompute affected shared results.

Measure how much reusable work actually exists, together with discovery, graph
construction/update, retained intermediate memory and repeated-call amortization.
Compare independent topology batching with explicit shared-subpattern execution
before choosing either. This is an investigation point, not a committed architecture.

## Staged investigation

1. Inventory real block paths from existing block-sparse tests and representative
   solver models. Record shape histograms, zero-path pruning, reuse, index counts,
   output fan-in and sequence dependencies. Include dynamic-interaction paths,
   special vertices, paired evaluation and correlators with right traversal.
2. Establish an allocation-bounded, library-backed device reference preserving
   existing CPU path/index semantics. Keep model/operator blocks resident and upload
   changed propagators explicitly. Reuse CPU/native-layout tests as independent
   checks; an all-dense comparison alone cannot detect identical path errors.
3. Build operation descriptors containing shapes, strides, operator/block IDs,
   lane/index provenance, data dependencies and output ownership, with bounded descriptor/cache/host/device memory accounting. Bucket independent
   ready operations by compatible shape/capability; use pointer-table/grouped or
   strided library batches where supported, measured small kernels elsewhere.
   Do not depend on a new cuBLAS API without checking the installed toolkit.
4. Test batching across indexed paths without evaluating zero blocks or padding
   every lane to the largest block. Charge bucketing, gather/scatter, metadata
   upload and dispatch to full API time. Compare a simple per-edge scheduler with
   reusable shape/path schedules before attempting broad fusion.
5. Fuse only compatible ready subpatterns (vertex/weight stages or bounded backbone
   segments). A size change, fan-in, library operation or synchronization requirement
   may form a fusion boundary. Preserve each lane's dependency order and caller's
   contribution order; grouping must not introduce atomic or tree-reduction rounding.
6. Measure heterogeneous mixtures, skewed shapes, changing paths, tails and different
   index counts. Include a case where small and library operations alternate inside
   one backbone. Run strict numerical checks, memory sanitizers and kernel/copy traces
   before deciding whether dispatch/fusion complexity earns its cost.

## Interfaces worth preserving during dense-small development

Separate matrix primitives from the square dense evaluator's traversal. Internally
carry explicit dimensions/strides and stable buffer ownership, with capability-tested
kernel selection. Keep rank transforms and cached LU distinct from operator products.
Allow immutable descriptors/pointer tables on one owned stream with explicit upload
handoff. Share validated primitives later, not the dense experiment's square storage
assumptions or whole-evaluator fallback logic. Do not compile every possible
rectangular `(M,N,K)` triple in advance; investigate a compact tile/family policy.

## Correctness and open questions

Retain zero-block semantics, subspace ordering, path multiplicity, parity/signs,
endpoint statistics and long indices. Preserve full complex-double arithmetic and
pivoted LU; the rejected custom solve demonstrates that tiny reconstruction residuals
are insufficient for high-order diagram agreement. Native owned-handle paths may support independent instances only after validation;
delegation to current dense/cuda retains its process-shared NDA serialization restriction.
A local adapter lock does not protect unrelated NDA users. Respect lane lifetimes during
asynchronous scheduling and avoid writes to metadata still in use.

Open questions: grouping opportunities after symmetry pruning; whether path discovery
is a significant CPU cost; packing/layout choices for variable widths; deterministic
fan-in accumulation; a bounded cache for changing shape sequences; graph usefulness
with heterogeneous schedules; and whether rectangular kernels justify extra compiled
variants. Resolve these through measured prototypes after dense-small integration.

Revisit and revise this block-sparse plan once the dense-small solver is operational,
using its validated primitives and new topology/path profiling evidence.

# Plan: an operational generic small-dimension CUDA evaluator

Date: 2026-10-12. Reviewed by GPT-6 Astra and revised after review and user clarification. Target:
`c++/triqs_xca/dense/cuda_small`, namespace `triqs_xca::dense::cuda_small`.
Keep `dense/cuda` as the library-based evaluator and retain all experiment folders,
source snapshots, measurements and failed controls. This task plans development;
it does not promote experiments or start a new implementation goal.

## Target hardware and orchestration scope

Optimize for high-end accelerators such as GH200, not the local RTX A4000. Existing
A4000 measurements are development evidence and regression controls; they do not
establish production cutoffs or target speedups. Qualify correctness and tune throughput,
capacity and crossover on target hardware before selecting defaults. If target access
is unavailable, mark dispatch choices provisional and record the missing validation.

Python is MPI orchestration, and its current API is not an optimization constraint.
Move work descriptions, chunking and ordered accumulation into native code as needed;
Python partitions work, assigns GPUs, submits bounded work and performs collectives.
Retain output/physics semantics while changing submission interfaces deliberately.

## Dense dimension scope

The current Solver is tailored to TRIQS `atom_diag` with N fermionic modes: the
full dense Hilbert-space dimension is d=2^N. Required matrix dimensions are therefore
2, 4, 8, 16, 32, ...; arbitrary positive dimensions are not a dense-solver requirement.
Below, existing benchmark labels such as N4 denote matrix dimension d=4, not four
fermionic modes. This restriction applies to Hilbert-space matrix dimensions only;
DLR rank, index count, active batch width and interaction-channel count remain runtime
quantities without a power-of-two requirement.

Use this discrete size set to simplify specialization and qualification. Symmetry
partitioning in the block-sparse solver produces varying block dimensions that need
not be powers of two, and operators between blocks may be rectangular. General
small/rectangular kernels and their dispatch belong to that later work; they are not
prerequisites for making the dense-small solver operational.

## Evidence and answers to the design questions

The accepted `fused_small` mode2 hardcodes4 in the custom products' dot loops and
row addressing, and16 in matrix sizes/output mapping. Vertex products use the same
helper. Pointwise fusion and cross-index batching are already dimension-generic;
rank transforms and pivoted LU remain library operations.

N4/r16,128 indexed ordinary-SE evaluations take6.8/9.2ms at orders6/8,14.0–14.8×
matching CPU in three processes. Single-index calls still favor CPU. E3 library-backed
batching costs32.7/44.5ms. The order6 profile attributes26.37ms to library tiny
products/vertices versus about0.40ms after custom kernels; remaining solves dominate.
Pointwise fusion alone gives under2%. These results justify a generic-small route,
but establish neither its dimension cutoff nor runtime-dimension overhead.

No runtime-N custom product has been benchmarked. There is therefore no measured
percentage loss to quote. Runtime loops/addressing could reduce unrolling and increase
instruction cost; they might also be close to templates once sufficiently batched.
Treat runtime-size kernels as an optional comparison, not a required dense feature.
Use same-run decomposition `T_new = T_other + gamma*T_products` to
understand whole-call sensitivity; microkernel slowdown is not evaluator slowdown.
Do not extrapolate the N4 speedup or grouped-FMA agreement to other sizes.

Preferred hypothesis: one shared runtime traversal/scheduler with a modest collection
of fixed-size product/vertex kernels for d=2/4/8/16/... and validated library fallback.
A runtime-size kernel is optional if measurements justify it; compiling the useful
power-of-two specializations is the primary route to evaluate.
Compile kernels, not complete evaluator pipelines for every N/rank/order/batch.
At construction select eligible kernel families/capabilities; at each call select
from these using method, rank and active width. No per-node CPU dispatch is needed
for a uniform dense call. Dimension is fixed per instance; rank, active lanes and topology remain
runtime values. Fixed-size specialization is feasible, but its benefit and membership
must be measured. A crossover is a surface in dimension/rank/lane count/method/GPU,
not one universal size. Required power-of-two dimensions outside the specialized set
use validated library fallback, subject to existing API and resource bounds. Supporting
arbitrary dimensions through lower-level FSet APIs is outside this solver milestone;
do not add synthetic non-power-of-two coverage as an acceptance requirement.

## 1. Extend the benchmark before choosing the specialization set

Preserve dense-small-v2 and its recorded data. Add a separately identified primitive
driver and extend physical fixtures as needed without changing recorded cases. The
current full-fermion fixture uses dimensions2/4/8/16 and derives orbital count from
dimension; this matches the intended dense-solver domain.

* Primitive coverage: square dimensions2/4/8/16, then32/64/... as feasible until a
  stable library-competitive region is observed on target hardware. Include complex
  noncommuting inputs, pointer-table vertices, beta0/1, supported alpha values,
  tails, strides and checked overlap rules. Arbitrary/awkward dimensions and rectangular
  kernels are deferred to block-sparse investigation.
* Evaluator coverage: physical full-fermion dimensions2/4/8/16 initially, extending
  through the measured crossover using bounded workloads and checked resource limits.
  Reuse atom_diag-based fixtures; no synthetic arbitrary-dimension FSet fixture is
  required. Retain independent low-order CPU/reference checks.
* Compare: retained production CUDA; E3 generic library batching; fixed power-of-two
  templates; and hybrid fixed-size/library dispatch. Optionally compare runtime-size
  or bounded-loop kernels at these same dimensions if this can simplify implementation
  without meaningful full-API loss. Equalize stream, schedule, LU, lane layout,
  precision and metadata policy so kernel choice is the changed variable.
* Sweep rank16/26/36 initially; index counts1/8/32/128 and capacity independently,
  B-1/B/B+1/2B+3 tails, changing-index/cache-miss calls, crossing/chain topologies,
  orders1/3/6/8 and bounded10/12. Use few deterministic indices at high order; bound
  total-index arithmetic for larger orbital counts rather than overflowing long.
* Record full API, constructor/first/cold-context costs, throughput, metadata/cache
  costs, owned/observed memory, register/local/shared memory, kernel/copy traces,
  compile time and binary size. Serialize GPU work; repeat promising dispatch decisions
  in three fresh processes with a second seed. Use matched explicit index lists.

Screen primitives and a few full-API cases first; expand coverage for promising
families rather than running the entire Cartesian sweep. Include a limited second
stage with varied pole and interaction-channel counts, which affect schedule size.

Numerical acceptance precedes speed: the existing gate is
`max_abs_error <= 1e-280 + 1e-10 * reference_norm`, individual contributions and aggregates, low-order CPU agreement and
separate high-order CPU diagnostics. Require finite inputs/results/norms/errors as
appropriate and explicit rejection of NaNs/infinities before comparisons. New dimensions can select different library
algorithms and reduction order. Preserve complex double, grouped real FMA/operand
order where validated; if a variant fails, reject it or use the validated library
path for that shape. Never add an absolute clamp or enable fast math to force passage.

Deliverable: checked variant/capability table and dimension/rank/lane/method crossover
measurements, with uncertainty and rejected variants. Select a small useful compiled
set from the power-of-two sequence through the measured crossover; no intervening
sizes need specializations. Prefer fixed-size kernels where they provide useful
full-API gains, and qualified library kernels elsewhere. If optional runtime kernels
are tested, consider them where performance is within measurement noise or a
provisional5% of a specialization and they reduce maintenance cost. The5% rule is a
proposed decision criterion, not a measured current result.

## 2. Design one common evaluator and replaceable primitive families

Initial files: `diagram_evaluator.hpp/.cpp`, `device_ops.hpp/.cpp`, `dlr_ops.hpp/.cpp`,
`kernels.cuh/.cu` and internal dispatch/configuration definitions. Keep the host header
surface compatible with existing dense evaluators and do not expose experimental modes
or std::function scheduling as the permanent public API.

Separate four policies: product kernel family; active lane/chunk width; workspace/cache
ownership; and public evaluator selection. Select a compact eligible kernel table at constructor
or workspace setup and call-time policy within it, with runtime call arguments for active batch/rank/strides. Dense
products are square, but internal descriptors should permit explicit M/N/K and layout
without assuming every future block-sparse vertex is square. Rectangular implementation
and heterogeneous scheduling belong to the later block-sparse plan.

Retain shared cached LU, wide RHS and rank GEMMs. Do not adopt `fused_solve`: it fails
strict public parity despite small residuals. Own stream/BLAS/solver handles and distinct
status storage, with explicit default-upload event handoff. Metadata and scratch must
outlive queued kernels. Document per-instance call serialization; independent-instance
concurrency needs a specific test, not reliance on shared NDA synchronization settings. The existing
`dense/cuda` fallback uses process-shared NDA solver state: delegated paths initially
require serialization with other NDA users, including native/delegated transitions.
An adapter-local lock cannot protect unrelated clients. Either document this process-
level restriction or give delegated paths owned solver state before claiming concurrent
facade instances. Test the selected contract explicitly.

Keep a validated generic library kernel family available inside the batched scheduler.
Compare it fairly with production `dense/cuda`, whose index loop is serial: a crossover
against E3 alone does not prove whole-evaluator fallback is preferable. Expose explicit
CPU/library-CUDA/small-CUDA selection first; automatic policy should use conservative
measured thresholds and avoid silently moving work to CPU.

## 3. Make batching and resource limits operational

Remove prototype limits as implementation details: order<=12, full enumeration<=4096,
384 weight arrays,64 pointer tables and256 status slots. Replace them with checked
sizes derived from topology/orbitals/endpoints and a bounded memory budget. Stream full
index enumeration into chunks without allocating the entire index list. Keep long
indices, invalid-prefix recovery, duplicates/multiplicity, empty calls and deterministic
caller-order output accumulation. Match legitimate baseline API bounds; this is not a
promise that arbitrarily large computations fit device memory.

Choose capacity using N/rank/available memory and workload; execute tails at an appropriate
width so129 indices do not require two full128-lane traversals. Measure small lane-family
choices versus runtime active widths. Use bounded content-keyed schedule ownership;
a small cache for multiple chunks must account for owned bytes and changed topology/
index contents. Re-upload changed G every call, regardless of identical host address.
Invalidate captured pointers after resize; operator/interaction snapshots change only
through documented reconstruction or explicit update semantics. Capacity shrink/retry
must be bounded, avoid partial-result double counting, and select only numerically
qualified widths. Account separately for host/device storage, pinned buffers, cached
metadata, retained delegated evaluators and temporary peaks during resize.

First preserve sequential output sums and library LU arithmetic. Later improvements
(ordered accumulation, staging reuse, graphs) require their own controls and regression
checks. Full API still includes transfers/materialization, not only traversal kernels.

## 4. Close the solver feature gaps

The prototype supports ordinary indexed SE only. Use
`dense/cuda/diagram_evaluator.hpp` and the CPU interface as the authoritative checklist:

|Missing functionality |Required implementation and validation |
|---|---|
|Full constructor/API compatibility |beta/eps/itops/FSet and real/complex atom_diag constructors; count overloads accepting Backbone/topology; dimension/rank/orbital/interaction getters and workspace accounting |
|Paired SE |all/scalar/vector overloads; both line directions, signs, zero-interaction pair restrictions and count semantics; preserve paired identities |
|Dynamic interactions |interaction-operator/coefficient constructor, fermionic/bosonic flavour bookkeeping, endpoint/operator choices, parity and mixed interaction paths; test density, spin-flip and rotated complex interactions |
|SPGF |all/scalar/vector and count methods, endpoint contractions and orbital output ordering; validate matching manual/reference routines |
|Correlators |eval_correlator all/scalar paths with fermionic flag; one-time correlator endpoint vectors/atom_diag/topology/indices; left/right traversal, reflection, statistics, rectangular endpoint collections and changed endpoint counts |
|Unrestricted legitimate calls |safe chunked enumeration/topology-derived workspace; validated ranks outside the tuned set via library operations; unsupported optimized paths get a correct explicit fallback |
|Actual Solver integration |Python bindings and exports, explicit evaluator selection in Solver.init_diagram_evaluator, documented default/backend configuration, bounded MPI work submissions and output semantics; CPU-only imports remain usable |

Feature-complete can initially mean optimized ordinary SE plus explicit delegation to
the existing CUDA evaluator for other methods, preserving full correctness and documenting
which paths are accelerated. Delegation costs/duplicate resident state count toward
setup/memory. Test transitions between methods and changed inputs. Final optimized coverage
can be delivered incrementally; never let an adapter running baseline methods masquerade
as tests of new small kernels for those methods.

Stage native paired/dynamic/SPGF/correlator paths using common primitives after the
compatibility facade works. Keep symmetry selection unchanged: dense selection does
not replace block_sparse for conserved-operator partitions. The current `scatter_range_over_ranks` materializes the full rank-local np.arange
for SE/pairs/SPGF/correlators; this must change. Prefer a native range/work-descriptor
API, or a begin/submit/finish accumulator that owns output across submissions, allowing
Python to send bounded ranges/chunks without storing all indices. Filter paired indices
inside the bounded work protocol. Do not sum partial host results per chunk without
qualifying the changed floating-point grouping. Preserve caller/index contribution order
inside native accumulation. Python must keep collectives outside uneven per-rank chunk
loops: finish local work, coordinate errors, then reduce. Test empty ranks, uneven and
filtered workloads, collective failure handling and explicit rank-to-GPU assignment.
Compare real self-consistency/observable runs, not only independent SE timing. No Python-accessible
CUDA evaluator is currently selected by the Solver, so C++ performance alone is not
an operational solver outcome.

### Backend configuration and feature-state visibility

Proposed orchestration setting: `dense_backend='cpu'|'cuda'|'cuda_small'`, with the
current CPU default initially. Explicit non-CPU dense selection with symmetry partitions
raises a clear configuration error; it does not disable symmetries automatically.
A future opt-in automatic policy can be added after target-hardware measurements.
Unavailable requested CUDA backends fail clearly at evaluator initialization, rather
than silently choosing CPU. Shape/method fallback inside an available cuda_small
backend remains explicit and reported.

Save backend preference and relevant policy settings in Solver copies/HDF5; update
constructor-key reconstruction rather than relying solely on __dict__.update. Older
archives without the new setting default to current behavior. Recreate handles/caches
from model data and never serialize GPU pointers/streams/evaluator state. Defer backend
availability checks until evaluation so archives remain inspectable on CPU-only machines.
Test reconstruction, old archives, changed model inputs and requested unavailable backends.
The precise Python signatures may change to serve the native work protocol.

Report native/delegated/unavailable capability per method at each milestone: M2
native ordinary SE with other paths not yet exposed; M3 full compatibility with explicit
fallback where needed; M4 native method coverage only after corresponding tests pass.
Operational completeness and acceleration coverage are separate acceptance criteria.

## 5. Integrate, validate and decide

Production currently globs `.cpp` and inherits NDA CUDA through TRIQS; custom `.cu`
needs conditional CUDA-language enablement and a separate thin kernel target. Adopt `XCA_CUDA_SMALL=AUTO|ON|OFF`: CPU-only keeps all CUDA unavailable; a CUDA-enabled
NDA build without a supported CUDA compiler still builds library CUDA (AUTO disables
small with a clear message); ON requires a usable small-kernel toolchain and otherwise
fails configuration; OFF preserves library-only CUDA. Export a distinct small-backend
availability definition. Build the custom target only when available, set architectures through a user-
configurable CMake policy rather than hardcoding86, and propagate static CUDA linkage,
PIC, install/export and Python binding dependencies. Keep C++23 host code separate from
minimal device headers as in the experiment. The CPU-only build must not require nvcc;
provide coherent unavailability diagnostics and no import failure. Binding generation
must follow actual backend availability, including checked-in generated bindings: use
conditional wrappers or a separate optional CUDA extension, and verify regeneration
and Update_Python_Bindings=OFF paths. Test CPU-only, CUDA-library-only and CUDA-small
configurations,
installed C++ consumers and Python imports. Preserve local configure scripts.

Reuse existing evaluator-parameterized analytic, unitary, dynamic-interaction, pair,
SPGF/correlator and indexing tests with the actual small evaluator for every claimed
native path. Add power-of-two primitive/physical-fixture coverage, dense complex bases,
rank/dimension/batch tails, empty/duplicate/changed inputs/endpoints, allocation failure,
limits, delayed uploads, cache invalidation and independent instances. Run GPU memory
checks and all applicable CPU/CUDA CTests; Python Solver tests must force each backend,
including ordinary/dynamic interactions, observables and representative self-consistency.

Milestones and exit conditions:

1. M1: power-of-two specialization/crossover experiments and reviewed dispatch table; no production edits.
2. M2: production cuda_small ordinary-SE primitives/traversal, resource handling and CPU-
   safe build integration; measured A4000 N4 regression gains retained, without requiring the same speedup on
   GH200; target performance measured separately and required power-of-two dimensions
   correctly supported by native kernels or validated library fallback.
3. M3: complete compatibility facade and explicit Python/Solver backend selection;
   fallback paths correctly tested, costs and native coverage reported.
4. M4: native paired/dynamic/SPGF/correlator coverage through the same small policies;
   full applicable C++/Python tests pass. Optimize only paths with measured benefit.
5. M5: installed builds and real solver workload comparison; choose documented defaults
   from stable evidence. All original experiments and the library evaluator remain.

Fallback uses a prequalified capability/dispatch table with tested dimension, rank,
lane width, method, precision and library/hardware environment restrictions. Inputs
and cancellation can matter beyond shape, so qualification is evidence, not universal
bitwise-equivalence proof. Unqualified combinations use a validated library route,
or the retained whole CUDA evaluator if batched-library arithmetic also fails. There
is no expensive per-call comparison against baseline in production. A failed numerical
qualification excludes the corresponding candidate configuration. An optimization with insufficient
full-API gain is omitted from dispatch even if its microkernel is faster. Missing target-hardware coverage is documented; only GH200-class measurements can
justify target defaults, and even those require policy keyed to the qualified environment. Preserve source/measurement
provenance, do not promote automatically, and keep plan/log commits separate from code.

## Astra review incorporated

Astra independently checked current kernels, public CUDA/FSet APIs, Solver MPI and
serialization code, build configuration and numerical harness. Revisions address
bounded Python work and MPI collectives, delegated shared-state concurrency, three
build-availability cases, backend persistence, finite-value/capability qualification
and call-time eligibility. Optional recommendations added: staged sweeps, bounded-loop
kernels, pole/channel coverage, full resource accounting and milestone capability labels.
User clarification makes Python submission API changes explicitly in scope and high-end
GH200-class hardware the target. Subsequent user clarification restricts required dense
Hilbert-space dimensions to powers of two from atom_diag, simplifying the specialization
set and removing arbitrary-size fixtures/runtime kernels as delivery requirements.
Varying non-power-of-two block dimensions remain in the block-sparse plan.
No implementation or benchmark run is claimed here.

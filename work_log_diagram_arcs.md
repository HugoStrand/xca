# Diagram arcs and binding review

Date: 2026-10-16. Branch: dev_backbone.

## Scope and motivation

The agreed change groups the choices attached to topology connections under
Diagram::arcs. Previously Indices was the input record, while Diagram retained
only its directions and poles and put its flavours into backbone vertices.
Correlator directions also changed layout during preparation. This made it
unnecessarily difficult to follow the same choices through the code.

ArcCatalog::hybridization_flavours() is unchanged: it deliberately returns only
the hybridization-family subset used for external single-particle operators.

## Implementation

Code commit: `b73990e` — `[diagram] group arc choices`.

- diagram/arcs.hpp replaces diagram/indices.hpp. Arcs owns directions,
  pole_indices and flavours. decode_arcs replaces decode_indices and returns
  the same record accepted by logical preparation.
- Directions always have one entry per topology row. Entry zero is -1 only for
  a correlator, whose external arc is absent. Pole and flavour slot a-1 belongs
  to internal topology row a, for a >= 1. The first self-energy arc still has a
  direction but no selected pole or flavour; its endpoints remain summed.
- Diagram owns Topology, Arcs, Backbone and whole-diagram prefactor information.
  It no longer has separate direction/pole fields. Input arrays are owned copies.
- prepare_vertices(topology, arcs, kind) now produces complete vertex
  prescriptions, including flavours and barred-operator pole slots.
  prepare_tau_kernels(topology, arcs, poles, kind) uses the same layout.
- Both preparation helpers check arc dimensions and direction conventions before
  indexing. Full preparation also checks model flavour bounds; kernel preparation
  checks numerical pole bounds. Prepared diagrams are consumed through const
  references; changing input choices requires preparation again.
- Dense and block-sparse evaluators, binding/support helpers, C++ tests and the
  backbone_cpu benchmark source use the grouped fields. The flat enumeration,
  range partitioning and Python APIs are unchanged. The C++ logical API changes
  deliberately: old compact correlator direction arrays are rejected.

No compatibility alias, second arc-selection type, inheritance, or GPU scheduling
policy was added. Numerical contraction, pruning and accumulation rules are
unchanged. Benchmark source is migrated for compilation only, not measured.

## Adversarial review

This is a source review of the resulting diagram and bind boundaries, including
existing limitations. The observations below are follow-up proposals, not claims
that the current solver call paths fail. They do not authorize a larger redesign.

### 1. Prepared-data consistency is still a caller contract

Evidence: diagram/diagram.hpp exposes both Arcs and its derived Backbone and
prefactor records as public mutable fields. Kernel placement depends on numerical
pole signs, but Diagram does not record the pole table that produced it. A caller
can change arcs.flavours without rebuilding vertices, or bind the prepared recipe
to a differently ordered or signed pole table. Existing evaluator flows prepare
from their own model and consume const records, so this is a reuse hazard rather
than an observed regression. Keeping the raw flavours adds one owned integer
vector per prepared diagram; no performance claim is made for that storage cost.

Recommendation: keep the plain records, document immutable use (done here), and
make preparation/model replacement the explicit invalidation boundary. If retained
cross-model plans become an API, add lightweight owner/revision checks at that
boundary. Avoid a cache framework or a hierarchy. Changing selected pole signs
requires re-preparation; changing only numerical tables while preserving the
logical prescription still requires fresh borrowed views.

Suggested validation: retain a prepared diagram across model replacement and
verify either explicit rejection or re-preparation matches a fresh evaluator.
Include a pole-sign change and a flavour reorder, not just a new propagator.

### 2. Binding accepts combinations it cannot verify

Evidence: bind::tau_kernels takes placement, prefactor and pole indices separately.
Its loops assume compatible extents and table indices. block_sparse::bind::backbone
checks range containment and metadata lengths, but not every sector transition or
recorded dimension against the supplied owners. bind::correlator indexes path and
operator spans using the completion record without checking index bounds.
Same-sized records from different discoveries can be incorrectly combined even
when simple dimensional checks pass. Numerical views also require their owners
to remain alive and unchanged; current comments state this requirement, but the
types do not enforce it.

Recommendation: add a normal bind::tau_kernels(diagram, values) entry point so
ordinary callers supply one coherent recipe. Keep component-level binding only
where needed. Establish one checked binding/preparation boundary for externally
retained records and a documented trusted internal path; do not repeat expensive
validation in every matrix operation. Table identity or revision checks belong
with retained batches, not with every scalar or matrix view.

Suggested validation: malformed kernel dimensions, out-of-range completion
indices, mismatched sector dimensions, and stale but same-shaped path tables.
The lifetime contract should be tested through owner replacement/rebinding,
not by deliberately dereferencing dangling views.

### 3. Endpoint preparation still partly lives in bind

Evidence: block_sparse/bind/self_energy.cpp::endpoint_sum enumerates symmetry-set
members and queries exact operator support to choose which endpoint terms exist.
It then attaches matrix/coefficient views. self_energy also assembles a discrete
diagram sign. A separate select_diagonal_self_energy step examines coefficient
values for nonclosing paths. Therefore the current SelfEnergyDiagram is a
candidate completion description, not a fully selected numerical work item.

Recommendation: first extract an owned, factored endpoint-term selection into
host preparation. Retain separate early and late lists rather than a Cartesian
pair expansion. Let binding attach values to that selection. Move representation-
independent sign rules into shared diagram helpers, preserving the explicit
reflection conventions. Exact coefficient support is numerical model metadata:
keep it outside abstract Topology/Arcs, and invalidate it when coefficients change.
Do not move a result-dependent cutoff into logical preparation.

Suggested validation: missing individual operators inside a symmetry set,
zero/nonzero nonclosing endpoint couplings, both external orientations and mixed
statistics. Compare complete outputs to the independent/manual routines.

### 4. Diagram kind and external completion are implicit after preparation

Evidence: Kind is supplied to preparation but not retained in Diagram. Consumers
assume self-energy or correlator semantics; the distinction is encoded indirectly
in direction zero and the number of backbone edges. External vertices use flavour
-1, and bind::backbone identifies the two unresolved locations from Topology.
Passing a prepared correlator into a self-energy entry point is not uniformly
rejected by a dedicated kind check.

Recommendation: a single retained Kind field with checks at public prepared-entry
points would be a small, understandable improvement. Keep the special endpoint
sum and correlator operators separate from the internal Arcs choices. Do not add
SelfEnergy/Correlator inheritance or a polymorphic diagram hierarchy.

Suggested validation: cross-kind prepared calls must reject cleanly, including
order one where there are no internal operators to expose the mismatch.

### 5. Host execution choices remain outside the shared records

Evidence: block_sparse/correlator_diagrams.cpp stores one beta path per starting
sector, consistent with the current single-target traversal. A future multi-target
operator expansion could have several such paths; the current lookup would retain
only the last. Discovery still obtains dimensions/group candidates from host
BlockOp objects, while the operands use nested vectors of host NDA views. Dense
operator binding also remains partly inside its evaluator methods.

Recommendation: retain the current host implementation while developing any
multiple-target or GPU executor as an explicit adapter. A multiple-target planner
needs a start-sector-to-list mapping and complete path-pair enumeration. For GPU
batching, pack the same logical choices, paths and endpoint terms into flat device
buffers with offsets; do not force host TauKernelViews or callback scheduling into
the shared records. Consider a small dense binding layer if the next dense executor
would otherwise duplicate operator selection. No common abstract executor is needed.

Suggested validation: synthetic branching support with two valid paths from one
start sector before enabling multi-target discovery. Measure allocation/binding
cost separately from arithmetic before introducing retained caches or packed
storage. Optimization should target high-end GPUs such as GH200; the local RTX
A4000 is only a development/qualification machine.

## Recommended follow-up order

First make kind and binding validation explicit. Next separate factored endpoint
selection from numerical binding and centralize shared sign rules. Only then
introduce retained batches, revision tracking or a device packing adapter if a
concrete execution experiment requires them. This preserves reviewable commits
and avoids replacing the current implementation with a speculative framework.

## Verification

- CPU reconfiguration and full build passed with binding regeneration enabled.
- All 74 CTest entries passed in 58.84 seconds, with OMP_NUM_THREADS=1,
  OPENBLAS_NUM_THREADS=1 and CTest parallelism 2. The indexing executable includes
  two new tests covering decoder/preparation layout, source and copy independence,
  first-order correlators, and invalid/mixed layouts at both preparation helpers.
  Existing independent diagram rules, sign tables, dense/block/manual comparisons,
  unequal block dimensions and Python tests all passed without tolerance changes.
- The existing range-selection Python check passed with three MPI ranks. The
  launcher printed the usual display-authorization notices and exited zero.
- The backbone_cpu target compiled and linked. No timing experiment was run.
- The Python tree, including generated wrappers, is unchanged from `5ea8de4`.
- git diff --check passed. ArcCatalog is unchanged. Initial unrelated worktree
  changes and untracked files were preserved.

Commands after `source ~/env/xca-cpu.sh`:

```bash
(cd cbuild && bash ../cmake.sh)
cmake --build cbuild --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild --parallel 2 --output-on-failure
cmake --build cbuild --target backbone_cpu --parallel 6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/cbuild/python:$PWD/cbuild/deps/adapol_src/src:$PWD/cbuild/deps/pyed_src:$PYTHONPATH"
mpiexec -n 3 python3 test/python/range_selection.py
```

Logs: `/tmp/xca-arcs-configure.log`, `/tmp/xca-arcs-build.log`,
`/tmp/xca-arcs-test.log`, `/tmp/xca-arcs-mpi.log`, and
`/tmp/xca-arcs-benchmark-build.log`. Temporary logs may disappear.

No regression was found in the changed supported execution paths. The review
above identifies contracts that would need strengthening for arbitrary retained
plans, model replacement, branching paths or a new GPU execution adapter; those
broader behaviors were not implemented or validated in this change.

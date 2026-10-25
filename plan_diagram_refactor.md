# Diagram and backbone refactoring

Agreed 2026-10-25 on `dev_backbone`, starting at `ad9e470`.
Progress, validation and code commits are recorded in `work_log_diagram_refactor.md`.

## Objective

Make the dense evaluator read as operations on a backbone, with explicit endpoint
handling. Preserve the functioning dense and block-sparse CPU evaluators and their
numerical conventions. This is a sequence of reviewable structural refactorings,
not a new execution framework or another numerical representation of Action.

## Agreed representation

- Keep Topology as the connectivity shared by many arc assignments. Its helper
  APIs remain public; do not isolate them or rewrite their independent tests.
- Represent Arcs as a vector of plain arc objects, one per topology row including
  zero. Each internal arc has direction, retained_flavour and pole_index. The
  special arc has invalid flavour/pole values; its direction is forward/backward
  for self-energy and explicitly external for a correlator. Validate this at
  preparation. No variants, inheritance or per-arc class hierarchy.
- Store one actual topology arc_idx on both endpoint vertices. Remove copied
  vertex flavour and both pole_slot fields. Retain barred/adjoint instructions.
  Flavour and pole are resolved through the associated arc; unbarred operator
  selection must not read an irrelevant pole.
- Backbone owns vertices, edges and the kernel-prefactor recipe. Each vertex
  carries its local kernel sign (zero is identity). Each edge owns an ordered
  list of {arc_idx, argument_sign} factors; empty means identity. Remove the
  parallel TauKernelPlacement container. Keep repeated factors and their order.
- Keep numerical TauKernel tables in Action. The prefactor remains a recipe,
  evaluated once for the complete backbone, including when two correlator ranges
  are used. A range does not own or automatically apply a copy of that prefactor.
- Diagram retains Kind explicitly and its existing internal direction_sign.
  PermutationSigns remains separate. Do not combine endpoint permutation,
  reflected-operator, expansion-order and kernel signs into one opaque scalar.
- Prepare the backbone coherently from topology, arcs, poles and kind. Keep
  simple value semantics: prepare again after changing inputs, then consume the
  prepared Diagram as const. No automatic invalidation/rebuild machinery.

## Explicit endpoint API

`Diagram::endpoint_ranges()` returns named `before_special` and `after_special`
BackboneRanges excluding vertex zero and its partner. Geometry derives from the
Topology and backbone extent, with no duplicated endpoint-index state.

`Diagram::self_energy_endpoint_sum(n)` returns the existing SelfEnergyEndpointSum
and rejects a correlator. It resolves the special arc's orientation centrally;
operator matrices and scalar coefficient tables remain in Action. Observable
operators remain explicit correlator inputs.

For order m and special vertex s, the ranges are:

- before_special: vertices [1,s), edges [0,s).
- after_special: vertices [s+1,2m), edges [s,edge_count).
- edge_count is 2m-1 for self-energy and 2m for a correlator.

Allow a well-defined empty self-energy tail when s=2m-1. Do not interpret a
vertex-only range as an empty range. Keep first-order correlator segments as
single-edge ranges. Review block-path consumers before relaxing range validation;
empty traversal must neither index a missing edge nor imply a nonzero contribution.

Dense self-energy evaluation traverses the initial segment, applies the endpoint
sum, then extends the result through the tail. Correlator evaluation computes the
two segments and joins them with observable operators. Shared geometry must not
force identical traversal or completion algorithms. Preserve paired evaluation's
opposite special-arc direction and its current range ownership convention.

## Reviewable implementation checkpoints

Each code checkpoint is followed by a separate `[docs]` work-log commit explaining
what changed, why, evidence and remaining work. Use terse existing commit style.
Do not mix plan/work-log changes into source commits, rewrite history or push.

1. **Integrate backbone kernels** (`[diagram] integrate backbone kernels`).
   Move the prefactor into Backbone, place vertex recipes with vertices and edge
   factors with edges, and update numerical kernel helpers and both evaluators.
   Introduce coherent backbone construction and remove obsolete placement storage.
   Temporary existing operator pole slots are acceptable until checkpoint 2;
   document that transition rather than changing every concept simultaneously.
2. **Unify arc indexing** (`[diagram] store arc choices together`).
   Introduce arc objects including the explicit special entry, migrate the flat
   decoder without changing its enumeration order, and replace vertex flavour /
   pole-slot duplication with arc_idx. Update operator selection, signs, support,
   kernel application, test fixtures and benchmark consumers together. Remove
   temporary representations from checkpoint 1.
3. **Expose endpoint geometry** (`[diagram] expose endpoint operations`).
   Retain Kind, support empty tails with precise range validation, add the named
   range and endpoint-sum APIs, and make dense traversal use them. Update sparse
   consumers where shared geometry is applicable, preserving their path planning.
   Reject wrong-kind numerical entry points before indexing or mutating buffers.
4. **Review and finish**.
   Review indexing, boundary treatment, arithmetic order, copying, construction
   dependencies and dense API readability. Fix concrete issues in a separate
   code commit if necessary. Record final tests and completion in the work log.

Checkpoint boundaries may be adjusted if compilation dependencies require it, but
keep a working evaluator at every code commit and explain adjustments in the log.

## Validation and review risks

- Preserve independent topology/parity tests, hand-derived indexing and sign
  tables, dense/block/manual numerical comparisons and existing tolerances.
- Tests reading old storage members need access-path changes, not wholesale
  replacement by expectations generated from the new implementation itself.
- Specifically cover first order, both arc directions, positive/negative/zero
  poles, mixed statistics, special arc at the final vertex, empty tails,
  reflected correlator traversal, pair ownership, copy independence and invalid
  arc metadata. Preserve edge-factor order/identity/repeated-pole tests, including
  existing overflow/underflow examples.
- Retained-flavour statistics assume parity-homogeneous F-bar contractions. This
  refactoring preserves that existing contract; it does not add support for
  arbitrary parity-mixing coefficients.
- Reconfigure when files move, build with the CPU environment and run appropriate
  focused checks at checkpoints. Final acceptance is a full CPU/Python CTest run
  with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and parallelism 2, plus compilation
  of backbone_cpu. Inspect regenerated bindings and git diff --check.
- Keep Python-facing signatures, flat-index ordering, numerical accumulation and
  paired-range ownership unchanged. Do not claim GPU validation or a speedup from
  this structural work. Extra edge-list allocations are a known tradeoff; packed
  storage can be considered later if measured preparation cost warrants it.

## Boundaries and future GPU work

No Python API redesign, MPI partition redesign, evaluator-overload cleanup,
block-path redesign, GPU implementation, benchmarking campaign, or Action/operator
container redesign is included. Do not move block-sector types into shared diagram
code. Plain logical descriptions must not require CPU materialization of every
indexed diagram/path before GPU execution. Future executors may enumerate work or
pack these descriptions differently. The eventual optimization target is high-end
hardware such as GH200, not specifically the local RTX A4000.

Preserve all unrelated worktree changes, local build scripts and dependency caches.

## Completion

Implemented and reviewed on 2026-10-25. Source checkpoints: `0f9c8fe`
(backbone kernels), `6e84f1a` (arc objects), and `91d5a68` (endpoint operations).
All 76 CPU/Python CTest entries pass on the final source; backbone_cpu compiles.
See work_log_diagram_refactor.md for per-checkpoint evidence, review findings and
remaining tradeoffs. No GPU performance or validation claim is part of this work.

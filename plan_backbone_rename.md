# Diagram and backbone naming plan

Implementation plan, started 2026-10-15; implemented and validated 2026-10-16 on `dev_backbone`.

The subsequent boundary review and implementation are recorded in
[work_log_backbone_boundaries.md](work_log_backbone_boundaries.md). The
**Shared backbone boundary correction** at the end of this document supersedes
the earlier placement of geometry, kernel prefactors and discovery APIs.

All six source stages are complete. See [work_log_backbone_rename.md](work_log_backbone_rename.md)
for the code-commit review index, separate documentation checkpoints, and validation
results. The final specification below governs the implementation; earlier sections
retain the naming discussion and superseded alternatives.

This document records the naming discussion following `plan_backbone.md` and
`work_log_backbone.md`. It separates agreed concepts from candidate names and
implementation questions. Update it as the discussion proceeds. Recording a
proposal here did not itself authorize implementation. The user explicitly added source
implementation and validation to the persistent task on 2026-10-16. The final
implementation specification below resolves earlier alternatives; no code changed
during the naming discussion.

## Purpose

Make four layers recognizable from names, namespaces and file locations:

1. Diagram concepts shared by dense and block-sparse evaluation.
2. The additional block-sector descriptions needed by block-sparse evaluation.
3. Numerical input data and bound operands.
4. The evaluators that execute the prescribed calculation.

Avoid unqualified “interaction” for pseudo-particle interactions: it is easily
confused with physical interaction terms, such as quartic products of physical
creation and annihilation operators. Use **arc** for the diagram-level concept.

## Agreed conceptual vocabulary

| Concept | Meaning |
| --- | --- |
| Backbone | Ordered sequence of operator vertices and backbone edges. |
| Backbone edge | A pseudo-particle propagator multiplied by its assigned scalar time-kernel factors. Its abstract prescription and numerical operands remain separate. |
| Arc | A pseudo-particle interaction connecting two backbone vertices. |
| Topology | Backbone ordering and arc connectivity, before selecting orientations, flavours and pole indices. |
| Diagram | Prepared diagram description: topology, selected indices, vertex and backbone-edge prescriptions, and associated signs. |
| Backbone segment | A specified interval of the backbone; the full interval covers the whole backbone. |
| Backbone path | A compatible sequence of Hilbert-space block sectors over a specified backbone interval, including its boundaries. Vertex operators connect sectors; the path determines propagator blocks and rectangular operator-block dimensions. |
| Operands | Numerical matrices, propagators, coefficients and kernel views bound for execution. |

A Diagram can still contain prescribed sums, such as barred-operator
contractions. Evaluation-specific completion prescriptions, such as the
self-energy endpoint sum or external correlator operators, sit alongside the
shared description. They need not be expanded. A Diagram does not select a
block-sector path; a single Diagram can admit several BackbonePaths.

Arc choices constrain the paths, but a BackbonePath follows the backbone, not
the arcs. Scalar time kernels can be assigned to vertices as well as backbone
edges; calling an edge a propagator-times-kernels product does not eliminate
vertex kernel factors.

The ordered backbone is largely implicit in vertex numbering today. There is
no requirement to introduce a standalone Backbone class merely to name it.

The external correlator operator pair is not a pseudo-particle interaction arc
and has no arc coefficient. The existing topology indexing convention can be
retained initially, but this distinction must remain explicit.

## Names settled in the discussion

Names below omit the outer `triqs_xca::` namespace.

| Current name/use | Intended name/use | Notes |
| --- | --- | --- |
| `DiagramTopology` | `diagram::Topology` | Connectivity, not a prepared diagram. |
| Raw topology input parameter | `topology_arr` | Explicit user preference; consistent with Python terminology. |
| `DiagramAssignment` | `diagram::Diagram` | Supersedes both the earlier `Backbone` and `Configuration` suggestions. |
| `AssignmentIndices` | `diagram::Indices` | Logical selections, independent of flat-index enumeration. |
| `prepare_assignment` | `prepare_diagram` | Function spelling follows the chosen Diagram name. |
| Common path information in `SelfEnergyPath` and `CorrelatorFragment` | `block_sparse::diagram::BackbonePath` | One concrete type for any backbone interval, including the whole backbone; evaluation-specific bookkeeping sits alongside it. |
| Backbone numerical inputs in self-energy and correlator structures | `block_sparse::BackboneOperands` | One concrete type for the selected blocks over that interval; no separate segment/fragment operand type. |
| `EndpointOperands` | `block_sparse::EndpointSumOperands` | Bound endpoint operators, coefficient views and endpoint statistics/sign information. |
| `CorrelatorJoin` | `block_sparse::CorrelatorOperands` | Replaces the proposed CorrelatorJoinOperands name with the same role; evaluated backbone results remain separate arguments. |
| Self-energy-specific block preparation metadata | `block_sparse::diagram::SelfEnergyDiagram` | Concrete block-specific description alongside the shared Diagram; exact member layout remains to be drafted. |
| Correlator-specific block preparation metadata | `block_sparse::diagram::CorrelatorDiagram` | Compatible backbone-path and external-operator choices; no bound matrix views or computed results. |

The desired distinction at a call site is:

```cpp
diagram::Topology topology{topology_arr};
auto prepared_diagram = diagram::prepare_diagram(topology, indices, /* ... */);
```

Illustrative only; signatures and local variable conventions remain to be
worked out. Avoid the current `DiagramTopology diagram{topology}` ambiguity.

## Folder and namespace direction

- `c++/triqs_xca/diagram/`, namespace `triqs_xca::diagram`: shared topology,
  logical indices, prepared Diagram, arc algebra/metadata, kernel placement and
  permutation-statistics concepts. Drop redundant `diagram_` file prefixes.
- `c++/triqs_xca/block_sparse/diagram/`, namespace
  `triqs_xca::block_sparse::diagram`: backbone paths, dimensions and
  diagram-specific block preparation metadata. Basic operator support can stay
  in `block_sparse`, since it is useful independently of diagrams.
- Dense/block-sparse numerical storage, operand binding and evaluators remain
  visibly distinct from those descriptions.
- User-proposed binding organization: `bind/` folders and `bind` namespaces,
  replacing function prefixes such as `bind_correlator_fragment` with qualified
  calls such as `bind::correlator_fragment`. See the binding discussion below.

This is a conceptual grouping, not permission to move every existing file
wholesale. Some files mix layers and need splitting. The precise home of shared
numerical arc data and kernel binding is still open.

Adopted: bring the existing root `topology.hpp/.cpp` utilities together with
Topology under `diagram/`. Retain a subordinate topology namespace if useful for
generation/manipulation functions; exact function placement remains to be listed.
Avoid two competing homes for topology concepts.

## Arc algebra and data

Adopted: use the arc vocabulary consistently in evaluator-facing C++ code,
including the ArcOrientation, ArcFamily, ArcCatalog and ArcData direction below.
Review public/Python consequences explicitly before changing exposed names.
Auxiliary operator-term names and exact data placement remain proposals.

| Current type | Candidate | Meaning |
| --- | --- | --- |
| `InteractionLine` | `diagram::ArcOrientation` | Direction-dependent endpoint adjoints, coefficient-index order, time reflection and reflection sign. Contains no topology endpoints. |
| `InteractionFamily` | `diagram::ArcFamily` | Flavour membership, physical family kind and statistics. |
| `InteractionModel` | `diagram::ArcCatalog` | Shared metadata describing available flavours/families; not the full physical model. |
| `hyb::Hybridization` | `ArcData` | Shared numerical pole, coefficient and time-function tables for hybridization and retarded arcs. Namespace/location undecided. |
| `OperatorReference` | `diagram::OperatorReference` | Flavour and adjoint flag, without matrix storage. |
| `WeightedOperator` | `diagram::WeightedOperatorTerm` | Operator reference, pole index, coefficient row/column and sign for one contraction term. |
| `BarredOperator` | `diagram::BarredOperatorContraction` | Prescription for summing the later flavour with earlier flavour and pole fixed. |
| `EndpointSum` | `diagram::SelfEnergyEndpointSum` | Special self-energy arc, with both endpoint flavours still summed. |

ArcCatalog and ArcData describe shared choices/tables used by many arc
instances. They are not separately owned by each topological arc. The name of
`InteractionKind` and its adapter functions remains to be decided consistently.

Current coupling, to preserve when organizing these types:

- Vertex preparation combines topology endpoints and InteractionLine conventions
  into vertex flags. It does not retain an InteractionLine object.
- Dense FSet and block-sparse BlockOpSymQuartet use BarredOperator terms to
  materialize barred matrices. A prepared diagram subsequently selects them.
- The barred contraction combines coefficients and operators. TauKernel
  placement supplies the corresponding scalar time dependence. Both use the
  selected pole index, but neither should absorb the other's responsibility.
- Dense self-energy execution uses EndpointSum directly. Block-sparse binding
  currently uses InteractionLine and support-restricted endpoint lists instead.
- InteractionModel supplies family/statistics metadata separately from this
  algebra. Permutation signs are distinct from direction/reflection signs.
- The special self-energy arc uses full time-dependent values, with no selected
  pole, unlike the internal barred-operator contractions.

## TauKernel placement, values and views

The discussion favors **TauKernel** over PoleKernel. The prefix is independent
of the placement/values/views separation. Pole emphasizes a parameter; Tau
emphasizes the scalar time function. These are proposed groupings, not existing
types under these names.

For notation, write K(t,x), where t is normalized imaginary time and x is the
beta-scaled pole. Plus/minus below selects the pole argument, not a fermionic
sign or the sign of the function value.

### `diagram::TauKernelPlacement`

- `vertices`: per-vertex internal-arc pole slot and argument sign (+1/-1/0).
- `edges`: per-backbone-edge and internal arc, argument sign (+1/-1/0).
- `prefactor_argument_signs`: select K(0,+x) or K(0,-x).
- `prefactor_exponents`: inverse powers of those zero-time factors.

The current KernelFactors helper also returns a discrete backward-arc sign.
Proposed separation: retain that with the Diagram's discrete signs, outside
kernel placement. Exact member names/layout are still open.

Important indexing distinction: current `pole_slot` indexes the Diagram's
selected pole-index vector, not the model-wide pole table. That slot also
selects a barred operator even when the vertex kernel sign is zero. Extracting
kernel placement must preserve the operator-selection role independently.

### `TauKernelValues`

Proposed named numerical member `kernel_values` within the current Hybridization
owner (or eventual ArcData). Ownership can remain there initially.

| Candidate member | Current member | Meaning |
| --- | --- | --- |
| `at_tau_plus` | `k_it_p` | Time-node/pole table K(t,+x). |
| `at_tau_minus` | `k_it_m` | Time-node/pole table K(t,-x). |
| `at_zero_plus` | `k_0_p` | Pole vector K(0,+x). |
| `at_zero_minus` | `k_0_m` | Pole vector K(0,-x). |

These tables depend on mesh and poles, not interaction coefficients or operator
matrices. The owner currently stores beta-scaled poles.

### `TauKernelViews`

Candidate replacement for BoundKernels:

- `vertices`: optional time-vector views for vertex factors.
- `reflected_vertices`: corresponding reflected-time factors; current binding
  selects the opposite pole argument.
- `edges`: a list of time-vector views per backbone edge, multiplied during
  execution rather than eagerly materialized as a product.
- `prefactor`: product of inverse zero-time kernel powers, excluding discrete
  diagram signs. Thus this object contains a scalar as well as views.

Illustrative binding boundary:

```cpp
auto kernel_views = bind_tau_kernels(
    prepared_diagram.kernel_placement,
    prepared_diagram.pole_indices,
    arc_data.kernel_values);
```

## Other proposed renamings to retain for discussion

| Current | Candidate / preference |
| --- | --- |
| `AssignmentSelection selection` | `FlatIndexSelection flat_indices`; keep flat enumeration at the API boundary. |
| `EndpointSigns signs` | `diagram::PermutationSigns permutation_signs`; distinguish from direction and other prefactors. |
| `ObservableBoundary boundary` | User prefers `CorrelatorStatistics` or `CorrelatorOperatorsStatistics`; shorter `diagram::CorrelatorStatistics` proposed. |
| DiagramAssignment member `poles` | `pole_indices`; values are indices, not pole values. |
| `SelfEnergyContribution` | `SelfEnergyOperands`, composed of BackboneOperands, EndpointSumOperands and evaluation bookkeeping. |

Names for vertex prescriptions, direction/orbital members, auxiliary preparation
records and local variables need a further consistency pass. Avoid silently
equating physical orbitals, arc flavours, symmetry sets and block sectors.

## Block-sparse structure agreed in the discussion

Revised on 2026-10-15 to consolidate whole-backbone and segment types. Use
ordinary concrete structs and composition. **Do not introduce abstract class
hierarchies, inheritance, or an extra generic evaluation framework.** Detailed
member layouts and remaining helper names are still proposals.

### One path type and one backbone operand type

`block_sparse::diagram::BackbonePath` describes:

- The backbone interval bounds.
- Its sector sequence, including boundary sectors.
- The corresponding dimensions required for execution.

The whole backbone is the full interval, not another type. There is no separate
BackboneSegmentPath, SelfEnergyBackbonePath or CorrelatorBackbonePath. Segment
remains useful vocabulary for the interval. The path does not need to know
whether a caller uses it for self-energy, [0,tau], or [tau,beta].

`block_sparse::BackboneOperands` supplies the borrowed propagator and resolved
vertex-operator blocks for that path/interval. Vertex selection must identify
operators supplied separately, including the special self-energy endpoints;
interval bounds alone do not encode that distinction. Numerical views borrow
live, unchanged owners as before. Kernel views currently remain separate inputs.

There is no separate BackboneSegmentOperands, BackboneFragmentOperands or
CorrelatorBackboneOperands. This consolidation requires a small structural
extraction from the existing types, not just text replacement.

### Where calculation-specific information belongs

Keep it alongside the shared objects, in concrete preparation/binding records
or explicit caller arguments as appropriate. Use SelfEnergyDiagram and
CorrelatorDiagram in block_sparse::diagram for the concrete block-specific
descriptions, as detailed below. Do not invent a hierarchy of diagram kinds.

| Layer | Shared description | Calculation-specific information alongside it |
| --- | --- | --- |
| Diagram prescription | Diagram: topology, selected arc indices, vertex/kernel prescriptions and shared signs | Self-energy endpoint-sum prescription, or external correlator operators/statistics |
| Block preparation | BackbonePath: interval, sectors and dimensions | Self-energy endpoint symmetry-set choices/output destination, or compatible correlator segment pairs/closing choices |
| Numerical binding | BackboneOperands: selected numerical blocks | EndpointSumOperands or CorrelatorOperands |

Extract the self-energy endpoint-set choices from the generic path. The output
sector is an accumulation destination even when equal to a path boundary; keep
that role explicit in evaluation bookkeeping. Flat index belongs at the selection
boundary, not in BackbonePath. Whole-diagram information such as the special
vertex is available from Diagram and should not be an intrinsic path field.

For the correlator, the [0,tau]/[tau,beta] role belongs to orchestration rather
than the generic path. Naming directions must not prescribe execution order.

### SelfEnergyDiagram and CorrelatorDiagram

These agreed names distinguish a block-specific realization from the shared
`triqs_xca::diagram::Diagram`. They describe one prepared contribution or
compatible path combination, not an eagerly expanded collection of all paths.
They are ordinary structs, not subclasses of Diagram or evaluators.

- SelfEnergyDiagram: a BackbonePath, endpoint symmetry-set choices, and output
  destination/completion metadata associated with the shared Diagram.
- CorrelatorDiagram: identities of a compatible pair of backbone paths, external
  operator indices and closing choices associated with the shared Diagram.
  Preserve the ability to reuse segment results and stream closing choices.
- Neither contains bound matrix views or computed backbone results. Binding
  produces the corresponding numerical operand records.

Agreed placement: `block_sparse::diagram::SelfEnergyDiagram` and
`block_sparse::diagram::CorrelatorDiagram`, under `block_sparse/diagram/`.
This follows the folder/namespace rule for abstract block-specific descriptions.
Exact member layouts and ownership details remain to be drafted.

Do not duplicate the shared Diagram for every streamed path merely to give these
records their names. The shared description can remain a separate immutable
input. Decide ownership of path metadata and references when drafting the actual
structs: no retained references into a reused visitor workspace. Correlator
records can identify retained paths/results without copying whole segment data.

This is the agreed placement for the already identified calculation-specific
metadata, not an extra wrapper around CorrelatorOperands. The decision that
CorrelatorOperands replaces CorrelatorJoin without owning two backbone inputs
continues to apply.

### Completion operands and execution

| Current structure | Intended structure/role |
| --- | --- |
| `SelfEnergyContribution` | `SelfEnergyOperands`: a simple aggregate of BackboneOperands, EndpointSumOperands, and overall sign/output bookkeeping as needed. |
| `EndpointOperands` | `EndpointSumOperands`: early/late operator blocks, coefficient views, and endpoint statistics/sign information. |
| `CorrelatorJoin` | `CorrelatorOperands`: external operator blocks and identifiers selecting the backbone results to combine, with closing-operator choices and observable indices. |
| `CorrelatorTrace` | Candidate name `CorrelatorClosureOperand`: one closing operator block and its observable index, not an evaluated trace. |

`EndpointOperand` can retain its existing name; the earlier proposal to prefix
every endpoint helper with SelfEnergy is not required by the chosen pattern.

**CorrelatorOperands replaces CorrelatorJoinOperands; it is not an additional
enclosing aggregate.** Preserve the existing CorrelatorJoin role. Two evaluated
backbone segment results are supplied separately to execution. It does not own
two BackboneOperands or their computed results. SelfEnergyOperands and
CorrelatorOperands need not have identical aggregation depth for naming symmetry.

The conceptual input pattern is:

- Self-energy: one BackboneOperands plus EndpointSumOperands.
- Correlator: two BackboneOperands used to produce segment results, followed by
  CorrelatorOperands to combine compatible results through external operators
  and take traces. Segment results can be reused across external-operator choices.

These are not identical execution sequences. The self-energy endpoint sum is
inserted partway through backbone evaluation, with remaining convolutions
afterward. The correlator combines two completed segment results. Splitting
input structures must not imply a new evaluation order.

Do not rename the self-energy-only aggregate DiagramOperands: that name would
hide its special endpoint sum and output handling. The general numerical
structure identified here is BackboneOperands. No DiagramOperands type is
currently planned.

### Remaining member-name proposals

`CorrelatorSide {right,left}` could become shared
`diagram::CorrelatorSegment {zero_to_tau,tau_to_beta}`. This identifies the
imaginary-time interval independently of how a product is drawn or evaluated.
It is not a block-sparse concept; dense correlator evaluation has the same
division. If retained, the label belongs to correlator orchestration rather than
BackbonePath. Exact interval representation and indexing remain to be designed;
preserve current traversal coverage, including valid single-edge segments.

- Path `propagator_blocks` -> `edge_sectors`: the values identify the sector
  carried by each backbone edge, not numerical blocks or storage offsets.
- Path `dimensions` -> `sector_dimensions`, with the existing index convention
  documented; do not imply that every array entry is an edge dimension.
- Self-energy `kap_set`/`mu_set` -> `early_symmetry_set`/`late_symmetry_set` in the
  calculation-specific preparation record, outside BackbonePath.
- Correlator result-selection start-sector names should follow the chosen
  segment vocabulary rather than left/right matrix placement.
- Keep numerical operand members named `propagators` while they hold bare
  propagator views. Rename to `edges` only if their type also describes the
  assigned kernels. The backbone-edge concept does not require eagerly
  materializing the propagator-times-kernels product.

### Support metadata

`OperatorSupport` already describes its role well: the directed sector
connections where an operator has a nonzero block. Keep that name.
`BlockConnection` -> `SectorTransition` and `OperatorSupports` ->
`OperatorSupportTable` are candidates to clarify its entries and collection.
The table includes plain, adjoint and barred-operator support; barred support
depends on the current coefficient data and must still be rebuilt after changes.

Support is reusable block-operator metadata, not itself a particular diagram or
backbone path. Prefer keeping the basic support types in `block_sparse` rather
than moving them under `block_sparse::diagram` solely because paths consume
them. The current table's Diagram-based vertex lookup and storage-reading
constructor require a boundary decision if stronger separation is desired.

### File organization and discovery

Candidate files: `block_sparse/diagram/backbone_path.hpp` for the common path
description; under the proposed bind organization,
`block_sparse/bind/backbone.hpp`, `block_sparse/bind/self_energy.hpp` and
`block_sparse/bind/correlator.hpp` for binding routines. Numerical operand-type
headers can remain in block_sparse and be included by bind routines; exact
inventory remains to be drafted.
Avoid presenting the current mixed `correlator_plan.hpp` as a single abstract
plan type: there is no such aggregate today.

Path discovery, binding and evaluation should have distinct verbs and types:
`visit_*_backbone_paths`, `bind::*`, and `eval_*`. Discovery currently reads
operator storage and propagator zero masks. Merely relocating its declarations
does not make it independent of numerical storage. Keep that adapter dependency
explicit; factoring it further is a structural change beyond a mechanical rename.

Keep preparation metadata separate from matrix views where useful, but do not
add join/closure wrappers solely for naming symmetry. Any extra concrete record
needs a clear role; it does not imply an abstract base class.

### Binding namespace discussion

There is no substantive objection to the user's `bind::correlator_fragment`
pattern. Prefer `triqs_xca::bind` for shared binding and
`triqs_xca::block_sparse::bind` for block-specific binding. Within block-sparse
code, `bind::backbone(...)` can handle either a segment or the full interval,
consistent with the consolidated BackboneOperands type. Function nouns should
follow the final concepts rather than preserve obsolete fragment distinctions.
Use explicit qualification when referring to shared bind routines from within
the block-specific namespace.

The important boundary is between **binding** and **owning numerical data**.
ArcData and TauKernelValues own input tables before any particular diagram is
bound; putting them in bind would blur that distinction. Agreed placement is
shared numerical-data headers outside diagram/ and bind/ (for example
`arc_data.hpp` and `tau_kernel_values.hpp`). TauKernelViews is a binding result
and can naturally be declared with the shared kernel-binding interface.

Binding selects and packages views; path discovery and matrix evaluation keep
their own names and responsibilities. No new binder class is implied.

## Adopted scope

Include the agreed structural separations along with renaming: common
BackbonePath/BackboneOperands, calculation-specific bookkeeping, and extraction
of TauKernelPlacement/TauKernelValues. Preserve algorithms, evaluation order and
numerical behavior. This is more than a mechanical rename, but not a new
evaluation architecture.

Adopt arc naming consistently in evaluator-facing C++, while auditing public
and Python-facing compatibility before renaming exposed symbols. Adopted default:
preserve existing Python-facing names where practical and explicitly list any
proposed changes. Consolidate topology utilities under diagram/. Numerical data
owners stay outside diagram/ and bind/; block-specific descriptions use
block_sparse::diagram. Prepare a concrete folder/type/member inventory before
implementation. The assistant can independently develop the reviewable commit
sequence and per-step validation.

## Agreed concrete layout choices

The user requests a simple design that will not force CPU orchestration or
storage choices onto a later GPU implementation and has adopted the following
recommendations. Exact member declarations and indexing examples remain to be
drafted within these agreed boundaries.

| Decision | Alternatives considered | Agreed choice |
| --- | --- | --- |
| Path metadata ownership | Own the small sector/dimension arrays; borrow walker workspace; use a shared arena | Own small arrays in retained records. Reuse discovery workspace, but explicitly copy/move retained metadata. No shared ownership framework. |
| Numerical input ownership | Copy matrices into each operand record; borrow views | Borrow const views from explicitly live input owners. Keep numerical buffers and evaluator scratch outside diagram descriptions. |
| Segment indexing | Full-diagram-sized arrays with inactive slots; compact interval arrays | Compact contiguous interval arrays with explicit bounds and documented mapping to diagram indices. Do not combine this with altered traversal semantics. |
| Correlator reuse | Copy path/result data into each pair; reference reusable entries by index | Explicit integer indices into caller-owned path/result collections, scoped to the current preparation/evaluation. No global cache or automatic invalidation mechanism. |
| Traversal order | Encode the current execution direction/schedule in the path; describe connectivity only | Describe connectivity in backbone order. Keep contraction direction, batching and scratch allocation in execution. |

Interval bounds alone do not specify whether boundary operators are included.
Document exactly which vertex/edge positions are covered and which endpoint
operators are supplied separately. Full self-energy and correlator segment
conventions differ here. Use explicit bounds/indices and simple data fields,
not overloaded sentinel meanings or a general expression/operation hierarchy.
Single-edge segments with no internal vertices remain valid. Concrete bounds
and layout must be checked against both current evaluator loops before adoption.

Keep dimensions readily available for execution, prepared from the authoritative
block dimensions. If stored redundantly in paths, establish consistency during
preparation/binding; do not leave discovery or shape lookup to the arithmetic
loop. Neither numerical views nor pointers into temporary visitor workspace
belong in SelfEnergyDiagram/CorrelatorDiagram.

### CPU/GPU separation to preserve

- Logical records use explicit indices, sectors, dimensions and coefficients'
  identities, independently of host/device matrix addresses. They prescribe
  mathematics, not a sequence of kernel launches or mandatory host callbacks.
- Existing CPU binding can produce NDA views. A future GPU binder can select
  device storage and pack bounded batches of descriptors. Do not make the CPU
  view-containing operand structs the required GPU transport format, or claim
  std::vector-owning records can be copied directly to device memory.
- Retaining a prepared record must not require immediate execution inside a
  discovery callback. Support streaming now and bounded accumulation later;
  do not require materializing all paths or endpoint-pair products.
- A compatible correlator pair refers to reusable segment results. Calculation
  metadata must not force recomputing either segment for every external operator.
- Input-dependent support/selection is resolved before numerical execution.
  It may need refreshing after input changes; the record is not an automatically
  valid cache. Avoid adding result-dependent host decisions to the executor.
- Preserve the present CPU algorithm/order during this refactoring. That is a
  validation boundary for this change, not a requirement that future GPU code
  retain the CPU accumulation order, contraction schedule or execution granularity.

No GPU execution framework, automatic scheduling, virtual classes, lazy
resolution, or new caching infrastructure is needed for these choices.

## Remaining implementation preparation and review constraints

No further architectural choice currently requires user input. The assistant
can resolve the concrete declarations, indexing conventions, helper names and
commit sequence below within the agreed design. Complete this preparation
before source implementation. Source work is now explicitly authorized by the
2026-10-16 request. Raise any discovered incompatibility that would materially change scope
or public behavior rather than silently changing the agreed boundaries.

1. Specify the common BackbonePath/BackboneOperands interval and boundary
   conventions, including which vertices are supplied separately. Move
   calculation-specific fields alongside them. Use concrete structs and
   composition, with no abstract class hierarchy or separate segment subtype.
2. Split abstract descriptions from numerical operands where files mix them.
   `block_sparse/correlator_plan.hpp`, for example, contains both fragment path
   metadata and operand/join types holding matrix views. Its visitors also
   currently consume numerical storage, so moving the whole header would not
   establish the intended boundary.
3. Finalize ArcData ownership/location and inventory the public/Python effects
   of renaming Hybridization. Track generated bindings and exposed names
   explicitly before deciding compatibility changes.
4. Draft concrete struct/member and file inventories for the adopted extraction
   scope, including SelfEnergyDiagram/CorrelatorDiagram and binding interfaces.
   A new vocabulary must not imply separation the code does not have.
5. Preserve numerical behavior, sign conventions, index ordering, borrowing
   lifetimes and range/vector API semantics. This discussion does not request
   new algorithms, benchmarking or CUDA changes.
6. Before implementation, group the work into reviewable, motivated commits:
   neither one large mechanical/structural rewrite nor many tiny noisy commits.
   Keep code commits separate from plan/work-log commits, following the existing
   terse commit style and recording what changed and why.
7. Include existing C++ tests, includes, CMake source discovery, bindings and
   benchmark callers in the eventual migration inventory. Tests deliberately
   exercise preparation APIs as well as numerical results; API updates must
   retain their independent expectations.

## Superseded suggestions

- `DiagramAssignment` -> `Backbone`: rejected by the clarified backbone meaning.
- `DiagramAssignment` -> `Configuration`: replaced by the agreed `Diagram`.
- Raw input `vertex_pairs`: replaced by the user's `topology_arr` preference.
- `ObservableStatistics`: use a correlator-specific name instead.
- Unqualified InteractionOrientation/InteractionData as the final vocabulary:
  superseded by the arc direction, to avoid physical-interaction ambiguity.
- “Propagator edge”: prefer backbone edge, explicitly including assigned kernels.
- Separate SelfEnergyBackbonePath/CorrelatorBackbonePath/BackboneSegmentPath:
  use one BackbonePath describing an interval, with calculation metadata outside.
- Separate CorrelatorBackboneOperands/BackboneSegmentOperands/
  BackboneFragmentOperands: use BackboneOperands for any interval.
- CorrelatorJoinOperands: replace with CorrelatorOperands, retaining the current
  external-operator/result-selection role. The suggestion to make it a complete
  aggregate containing both backbone inputs was explicitly withdrawn.
- DiagramOperands for SelfEnergyContribution: too broad for that structure;
  retain SelfEnergyOperands and extract the shared BackboneOperands portion.

## Discussion log

- 2026-10-15: Recorded the accumulated discussion through the agreement on
  BackbonePath. No source edits or renaming commits made. Continue revising this
  document as names and boundaries are settled.
- 2026-10-15: Added block-sparse proposals after checking the actual self-energy
  paths, correlator segment traversal, joins, support metadata and numerical
  binding. Distinguished path descriptions from join/closure operands and kept
  the proposed names explicitly provisional.
- 2026-10-15: Reconciled subsequent decisions: one interval-based BackbonePath
  and BackboneOperands, concrete composition without class hierarchies,
  calculation-specific metadata alongside the shared structures,
  EndpointSumOperands, and CorrelatorOperands replacing the join name without
  adding an enclosing type. Recorded the different self-energy/correlator
  execution sequences. Updated earlier tables and file proposals accordingly.
- 2026-10-15: Recorded adoption of structural extraction with unchanged numerical
  behavior, consistent evaluator-facing arc naming with a public/Python audit,
  and topology consolidation. Added the user's bind namespace and
  SelfEnergyDiagram/CorrelatorDiagram proposals, with binding-versus-storage and
  namespace/ownership distinctions left explicit for discussion.
- 2026-10-15: Confirmed SelfEnergyDiagram and CorrelatorDiagram in namespace
  block_sparse::diagram, matching the description folder. Namespace choice is
  settled; concrete member layouts and ownership details remain to be drafted.
- 2026-10-15: Confirmed numerical owners outside diagram/bind and the default of
  preserving Python-facing names where practical. The assistant will develop
  commit sequencing/validation. Added concrete ownership/indexing options and
  recommendations, with explicit boundaries to preserve future GPU batching,
  device binding and independent execution schedules without a new framework.
- 2026-10-15: User adopted the concrete ownership, compact indexing, correlator
  reuse and execution-independent path recommendations. Marked these as agreed;
  remaining work is detailed implementation preparation within those boundaries.

## Implementation specification (2026-10-16)

This section resolves the earlier member/layout proposals. It is authoritative
where the discussion record above still offers alternatives. The user has now
explicitly authorized source implementation, validation and the established
separate code/documentation commit pattern on `dev_backbone`. Preserve unrelated
working-tree changes. No push, history rewrite, CUDA port or cluster run belongs
to this task. Future GPU targets remain high-end devices such as GH200, not tuning
specifically for the local RTX A4000.

### Shared types and files

All paths below are relative to `c++/triqs_xca/`. Header/source pairs are implied
where there is non-inline implementation. Keep the existing namespace indentation
and ordinary NDA/std::vector value types; no type-erased tasks or inheritance.

| Existing file/type | Final location/name |
| --- | --- |
| `diagram_assignment.*`, DiagramTopology | `diagram/topology.*`, `diagram::Topology` |
| Root `topology.*` utilities | Same `diagram/topology.*`, functions in `diagram::topology` |
| DiagramAssignment, VertexAssignment | `diagram/diagram.*`, `diagram::Diagram`, `diagram::Vertex` |
| AssignmentIndices, DiagramKind, decode/count helpers | `diagram/indices.*`, `diagram::Indices`, `diagram::Kind`, `decode_indices`, `count_diagrams` |
| `diagram_selection.hpp` | `flat_index_selection.hpp`, `FlatIndexSelection` at the compatibility boundary |
| `diagram_model.*` | `diagram/arc_catalog.*`, ArcKind/ArcFamily/ArcCatalog and `hybridization_retarded_catalog` |
| ObservableBoundary/uniform_observable | `diagram/correlator.hpp`, CorrelatorStatistics/`uniform_correlator_statistics`; CorrelatorSegment labels here too |
| `interaction.hpp` | `diagram/arc.hpp`: ArcOrientation, OperatorReference, WeightedOperatorTerm, BarredOperatorContraction, SelfEnergyEndpointSum |
| `diagram_sign.*` | `diagram/sign.*`: PermutationSigns, permutation_sign, prepare_permutation_signs, Statistics (or included catalog definition) |
| KernelFactors/VertexKernel | `diagram/tau_kernel.*`: TauKernelPlacement, VertexTauKernel; discrete direction sign remains in Diagram |
| Hybridization owner in `hyb.*` | `arc_data.*`, `triqs_xca::ArcData`, with a TauKernelValues member |
| Scalar table owner | `tau_kernel_values.hpp`, `triqs_xca::TauKernelValues` |
| `kernel_binding.*` | `bind/tau_kernel.*`: `bind::tau_kernels`, TauKernelViews and aliases |
| apply_kernel/apply_kernel_product | `tau_kernel_ops.*`, ordinary arithmetic helpers outside bind |

Retain `hyb.*` for the physical coefficient conversion/extension utilities for
now: `hyb::coefs2vals` and `hyb::get_extended_coefficients` have legitimate existing
callers and need not be renamed just because the storage owner changes. ArcData
may call these helpers. Do not touch the legacy triqs_soehyb solver.

Concrete shared member sketch (constructors/validation omitted):

```cpp
namespace triqs_xca::diagram {
struct Indices {
  nda::vector<int> directions, pole_indices, flavours;
};
struct Vertex {
  bool barred, adjoint;
  int flavour;   // -1 only for the two separately supplied special vertices
  int pole_slot; // internal-arc slot; meaningful for a barred operator
};
struct VertexTauKernel { int pole_slot; int argument_sign; }; // 0 means no factor
struct TauKernelPlacement {
  std::vector<VertexTauKernel> vertices;
  nda::array<int, 2> edges;
  nda::vector<int> prefactor_argument_signs, prefactor_exponents;
};
struct Diagram {
  Topology topology;
  std::vector<Vertex> vertices;
  nda::vector<int> directions; // full arc order; absent correlator arc remains -1
  nda::vector<int> pole_indices;
  TauKernelPlacement kernel_placement;
  int direction_sign;
};
}
```

Diagram owns its arrays as today. Topology owns `arcs`; retain that name for the
endpoint-pair array and document row zero's external-pair convention. Diagram
has no flat index; flat adapters decode and prepare, retaining the index only in
caller diagnostics if needed. Do not add another generic selection wrapper type.
The tests for flat identity become tests of decoding/counting and independent
prepared values, not requirements to store the enumeration index in a Diagram.

`prepare_diagram(topology, flavours, indices, poles, kind)` remains the logical
entry; a flat-index overload is a convenience adapter. Move/rename the existing
preparation, rather than inventing a new algorithm. ArcCatalog keeps its current
validated family/statistics/external-hybridization lists and accessor semantics.
CorrelatorStatistics has `at_zero` and `at_tau`; other signs remain separate.

ArcData keeps tau_mesh, beta-scaled poles, coeffs, values and values_reflect plus
`kernel_values`. Constructor pole arguments remain physical frequencies. The four
TauKernelValues fields are at_tau_plus/minus and at_zero_plus/minus, as specified
above. Preserve existing reflection-sign initialization, notably the dense and
block-sparse callers' different conventions; do not apply the reflection minus
twice. Kernel binding uses pole slots and selected pole indices without touching
coefficient or operator matrices. `bind::tau_kernels` returns vertices,
reflected_vertices, edges and prefactor; numerical factors remain ordered as now.

### Backbone intervals: explicit vertex and edge bounds

Use half-open bounds. With n=2m vertices, label sector slot s_i immediately
before vertex i; vertex i maps s_i to s_(i+1). Edge e carries s_(e+1). The
correlator has an extra trailing edge e=n-1 to beta; self-energy has edges
0 through n-2. The shared kernel placement has n-1 edge rows: that final
correlator edge has no internal-arc kernel factors. Never index a nonexistent
kernel row when binding or executing it.

```cpp
namespace triqs_xca::block_sparse::diagram {
struct BackbonePath {
  int first_vertex, vertex_end;
  int first_edge, edge_end;
  nda::vector<int> sectors;
  nda::vector<int> sector_dimensions;
  // Local slot i corresponds to global sector slot first_vertex+i.
  // sizes are vertex_end-first_vertex+1, including both boundary slots.
  // Accessors sector(i), dimension(i), edge_sector(e) do only index translation.
};
struct SelfEnergyDiagram {
  BackbonePath path;
  int early_symmetry_set, late_symmetry_set;
  int output_sector;
};
struct CorrelatorDiagram {
  int zero_to_tau_path, tau_to_beta_path; // indices in retained path/result tables
  int at_tau_operator;
  std::vector<int> at_zero_operators; // compatible closing choices, current order
};
}
```

No duplicated edge_sectors array is needed: edge_sector(e) is
sectors[e+1-first_vertex]. Both vertex and edge bounds are explicit because
self-energy includes its outer vertices whereas correlator segments exclude the
external operators. Validate these relationships in preparation/binding, not
repeatedly in arithmetic. There are no invalid-sector placeholders in a retained
valid path. A discovery workspace may still use an absent value internally.

| Use | Vertex interval | Edge interval | Sector slots stored |
| --- | --- | --- | --- |
| Self-energy | [0,n) | [0,n-1) | s_0 through s_n |
| Correlator zero-to-tau, special vertex j | [1,j) | [0,j) | s_1 through s_j |
| Correlator tau-to-beta | [j+1,n) | [j,n) | s_(j+1) through s_n |
| Any single-edge segment e | [e+1,e+1) | [e,e+1) | s_(e+1) only |

For crossing order two, n=4,j=2: full self-energy stores five sectors and three
edges; correlator segments each store two sectors and two edges, with vertices
1 and 3 respectively. For first order n=2,j=1, both correlator segments have no
internal vertices and one edge. Closure connects their boundary sectors through
the external operators; it is not an equality requirement on each segment.
SelfEnergyDiagram's output sector equals path.sector(0) in this implementation;
path.sector(n) is the terminal sector and is not assumed equal during discovery.
Keep the existing input-coefficient closure validation before arithmetic.

### Numerical operands and binding

Keep operand records under block_sparse, not block_sparse::diagram. A plain
BackboneOperands owns its small path metadata and a compact vector of propagator
views, indexed by e-first_edge. Its vertices are a compact interval-sized vector
of optional matrix views, indexed by v-first_vertex. An empty optional means
exactly 'supplied by completion operands': self-energy vertices 0 and j. All
correlator internal vertices have a view. This is one type for either interval,
with no global-diagram-sized padding and no generic operation tags. Existing
arithmetic must never request an absent vertex; no support search is hidden in
an accessor. Bounds/accessors provide direct indexing.

```cpp
struct BackboneOperands {
  block_sparse::diagram::BackbonePath path;
  std::vector<std::optional<MatrixConstView>> vertices;
  std::vector<PropagatorConstView> propagators;
};
struct EndpointSumOperands {
  std::vector<EndpointOperand> early, late;
  std::vector<CoefficientConstView> coefficients;
  int reflection_sign;
  triqs_xca::diagram::PermutationSigns signs;
};
struct SelfEnergyOperands {
  BackboneOperands backbone;
  EndpointSumOperands endpoints;
  int output_sector;
  int sign;
};
struct CorrelatorClosureOperand { int at_zero_operator; MatrixConstView matrix; };
struct CorrelatorOperands {
  int zero_to_tau_path, tau_to_beta_path;
  int at_tau_operator;
  MatrixConstView at_tau_matrix;
  std::vector<CorrelatorClosureOperand> closures;
};
```

MatrixConstView, PropagatorConstView and CoefficientConstView above abbreviate
the current NDA const-view types, not a proposed abstraction framework. Keep
EndpointOperand's matrix/coefficient_index/statistics fields. Copying an operand
record owns metadata but borrows numerical values; owners and storage addresses
must remain valid and unchanged until work completes. Async GPU execution will
require ownership through completion, not merely until launch. Rebind after any
relevant replacement; no automatic cache generation/invalidation machinery.

Use `block_sparse/bind/backbone.*`, `self_energy.*`, `correlator.*`, `vertex.*`
for functions in block_sparse::bind. Put common operand declarations in
`block_sparse/backbone_operands.hpp`, `self_energy_operands.hpp`, and
`correlator_operands.hpp`. Bind functions accept the shared Diagram explicitly,
then the block description/path and actual numerical owners. `bind::backbone`
handles either interval; `bind::endpoint_sum` uses SelfEnergyDiagram;
`bind::self_energy` composes those results; `bind::correlator` supplies only the
external operators, while arithmetic receives evaluated segments separately.
Shared kernel binding is explicitly `triqs_xca::bind::tau_kernels` within a
block_sparse::bind scope to avoid accidental namespace lookup.

Place self-energy discovery in `block_sparse/diagram/self_energy.*` and
correlator discovery/completion metadata in `block_sparse/diagram/correlator.*`.
These are host preparation adapters reading operator connectivity and propagator
masks. Their returned records are owned numerical-storage-independent metadata;
do not claim the discovery code itself is independent of these input adapters.
Keep OperatorSupport and the renamed OperatorSupportTable/SectorTransition in
block_sparse/operator_support.*. Exact barred support still needs rebuilding
when coefficients/operators change. Multitarget execution remains unsupported.

Correlator preparation retains compact path tables for the current shared
Diagram and maps start sectors to table indices. Separate result tables have the
same indices. Preparation emits CorrelatorDiagram records in the current
mu/start/kap order, extracting source sectors from retained paths at binding.
Do not introduce a Cartesian table of all joins or a persistent cross-assignment
cache. Retained diagrams contain indices, never references to reused workspace.
Mask checks and exact-vs-candidate operator support behavior stay unchanged.

### Public and repository migration inventory

The two evaluator class names, raw topology/scalar/vector/range overloads,
constructors, output shapes, and Python module exports remain unchanged.
C2PY_IGNORE excludes prepared types and ArcData members; `_triqs_xca.toml`
explicitly selects only the existing evaluator aliases/free functions. Regenerate
bindings and inspect the result rather than assuming all transitive headers are
invisible to clair-c2py. Keep the constructor/method extern templates in
`python/triqs_xca/_triqs_xca.cpp` and the public Python names unchanged.

A concrete compatibility exception: generated bindings currently expose the
keyword `topology`, as well as `hyb_poles`, `hyb_coeffs`, etc. Keep those names in
Python-facing C++ declarations; use topology_arr in internal/prepared helpers.
Do not silently rename Python keyword arguments to obtain uniform C++ spelling.
Keep existing public `get_num_*_backbones` names as boundary compatibility APIs,
although internal count helpers use diagrams. No new Python wrapper types.

Update all C++ consumers: dense/block-sparse evaluators, FSet/BlockOpSymQuartet,
manual/reference routines that include moved headers, shared test utilities,
preparation tests, and benchmarks/backbone_cpu/main.cpp. Do not rewrite historical
CSV results or work logs. Build the EXCLUDE_FROM_ALL backbone_cpu target explicitly
as a compile check; timing experiments are not needed for this refactor.

CMake's xca_add_library uses recursive *.cpp discovery and installs headers
recursively; reconfigure after moves so old translation units disappear and new
ones compile. Avoid duplicate definitions from old/new copies. Current tests are
also discovered recursively. Keep test file names initially to make comparison
simple, updating includes/types and meaningful ownership/index assertions.

This is a source-level C++ API migration on dev_backbone, not an ABI-compatible
release. Prefer direct migration of in-tree consumers to a permanent web of
legacy aliases. Python compatibility is deliberately preserved. Legacy solver
sources, user experiments, configure scripts, cached dependencies and unrelated
working-tree files are outside the change.

### Commit sequence and checkpoints

Each row is one reviewable code commit (split only if validation exposes a
substantial independent issue), followed by a separate terse [docs] commit updating
`work_log_backbone_rename.md`. The initial completed plan and work-log checkpoint
are committed before source edits. Do not squash or push. All commits stay on
current dev_backbone. Every code commit should build; temporary compatibility
scaffolding must be identified and removed before final acceptance.

| Step / code subject | Motivation and scope | Focused validation |
| --- | --- | --- |
| 1. `[diagram] group shared descriptions` | Move topology/assignment/model/sign/arc concepts under diagram; apply the agreed names, consolidate topology helpers and update consumers. Keep existing layouts during this mechanical step. | Full build; topology, backbone_indexing, backbone_parity, interaction_contractions, both prepared_evaluator tests. |
| 2. `[diagram] separate arc data and kernels` | ArcData/TauKernelValues ownership, explicit placement versus vertex selection, bind namespace and kernel arithmetic separation. Remove flat identity from Diagram and keep decoding at the boundary. | hyb, indexing/parity/contractions, dense and block-sparse prepared evaluation, dynint/correlator tests; inspect generated wrapper keywords. |
| 3. `[block_sparse] describe backbone intervals` | Implement the owned compact BackbonePath and SelfEnergyDiagram; migrate both discovery paths while preserving masks/order. Keep completion/side information outside generic paths. | self_energy_path, correlator_plan, numerical_zero, sym_set_mask, retained-path ownership tests. |
| 4. `[block_sparse] bind backbone operands` | Common compact BackboneOperands, EndpointSumOperands, SelfEnergyOperands and block bind functions; preserve endpoint insertion and convolution order. | prepared_evaluator, sparsity_invariance, evaluator_api, dynint_evaluator plus dense reference comparisons. |
| 5. `[block_sparse] separate correlator preparation` | Explicit CorrelatorDiagram metadata and CorrelatorOperands; retained path/result indices, segment reuse and external-operator binding. | correlator_plan, one_time_correlator_all_evals, dense/correlator_statistics, mixed-statistics/one-edge cases. |
| 6. `[diagram] finish evaluator naming` | Remove transitional names/includes; align local topology/diagram/arc/kernel names and documentation, support helper names, benchmark compile caller, public compatibility audit. | Full CPU CTest, 3-rank range_selection check, wrapper/export audit, explicit backbone_cpu target build. |

If stage 3 changes a path layout consumed by stage 4/5 code, include the minimal
consumer indexing adjustments in stage 3; defer operand-structure extraction to
the next stage. Never keep duplicated algorithms merely to make commits appear
mechanical. A work-log entry states the relevant old/new concept, rationale,
files, validation and any temporary adaptation, with its code commit hash.

### Validation and review criteria

Use the existing environment/configure scripts and preserve cached dependency
checkouts. Before source changes record a CPU build/test baseline. Configure
with `source ~/env/xca-cpu.sh; (cd cbuild && bash ../cmake.sh)` in ordinary shell
steps, build with `cmake --build cbuild --parallel 6`, and run CTest with
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1. Full acceptance is the current complete
suite, not the older 61-test count in AGENTS.md. The last recorded baseline was
73 entries; discover the current count rather than invent a fixed target.

Add/adjust tests only for new structural contracts: independent retained path
copies, compact sector indexing on both intervals, self-energy external boundary
sectors, trailing correlator edge, and vertex/pole-slot separation when no vertex
kernel exists. Preserve hand-written expected values and independent manual /
all-evaluator comparisons. Do not replace numerical checks with two aliases of
the same implementation. Flat-index tests continue to check 64-bit boundaries,
overflow, decode ordering, logical preparation beyond representable flat counts,
empty ranges, and dense paired-range behavior; storage of the flat index itself
is intentionally removed.

CPU evaluation acceptance includes all existing exact-support, barred-update,
nonclosing-path, zero-mask, mixed-statistics, rotated-operator and arbitrary
propagator-layout tests. Preserve sum/prefactor placement and retarded reflection
signs. Public Python serial/MPI range semantics and imported evaluator names must
survive regeneration. Use the existing cbuild test environment for the MPI test;
record its exact invocation in the log. No GPU benchmark or Arrhenius allocation
is necessary; this task's GPU criterion is independence of the logical records,
not a claimed measured speedup.

Adversarial review of the specification resolved these traps:

1. Edge-only bounds lost outer self-energy sectors and the trailing correlator
   propagator: explicit vertex/edge intervals and sector-slot examples fix both.
2. A renamed vertex pole slot could be dropped when its kernel sign is zero:
   retain operator selection separately from TauKernelPlacement.
3. Copying borrowed callback data could retain invalid references: retained
   metadata owns arrays; correlator relationships use caller-scoped indices.
4. Namespace diagram shadows the shared namespace inside block_sparse: use
   fully qualified shared types or selected using declarations, never broad
   `using namespace` to conceal this collision.
5. A new Python keyword topology_arr would break keyword calls: preserve the
   boundary spelling and use the new spelling internally.
6. A whole CorrelatorOperands aggregate would undo the user's clarification:
   it binds external operators only and takes computed segment results separately.
7. A callback-only executor or host NDA view format would constrain GPU work:
   owned logical records can be retained and device binding is a distinct adapter.
8. Self-energy completion is not a postprocessing join: keep its existing
   insertion point and remaining convolutions.

Implementation completion: all six stages are committed, with separate work-log
checkpoints. Full CPU build and 73/73 CTest entries passed, as did the three-rank
MPI range-selection check and explicit backbone_cpu target build. Generated Python
bindings differ from the pre-implementation checkpoint only in two class docstrings.
See work_log_backbone_rename.md for code hashes, commands and remaining limitations.

## Shared backbone boundary correction (2026-10-16)

The adversarial review found that folder separation alone had left common
backbone geometry in block-specific records. The user authorized the following
correction, implemented as three further code/documentation checkpoints.

| Concept | Location and responsibility |
| --- | --- |
| BackboneRange | `diagram/backbone_range.hpp`: validated half-open vertex and edge bounds, independent of sectors and evaluation direction. |
| Backbone | `diagram/backbone.hpp`: owns ordered operator vertices and local TauKernelPlacement; owned by Diagram. |
| TauKernelPrefactor | `diagram/tau_kernel.hpp`: whole-diagram scalar recipe, owned by Diagram alongside direction_sign. |
| BackbonePath | `block_sparse/diagram/backbone_path.hpp`: composes BackboneRange and adds owned sectors and dimensions. |
| SelfEnergyDiagram / CorrelatorDiagram | Plain block-specific realization/completion records under `block_sparse/diagram/`; no host discovery declarations. |
| Host discovery | `block_sparse/backbone_paths.*`, `self_energy_paths.*`, `correlator_diagrams.*`, with functions in `block_sparse`. |

Shared `correlator_backbone_ranges(topology)` returns the zero-to-tau and
tau-to-beta ranges. Dense evaluation and block discovery use the same geometry.
CorrelatorSegment is removed: discovery takes the range itself. Block completion
still records compatible sector-path pairs and external-operator choices; a
shared range does not replace that block-specific information.

Backbone includes every propagator edge. Self-energy has 2m-1 edges; a correlator
has 2m, including the final edge to beta with an empty scalar-kernel list. This
supersedes the earlier n-1-row placement convention for correlators. Block binding
checks the range against the actual prepared backbone extent.

CPU forward/backward traversal routines consume bound backbone operands and
kernel views. Correlator orchestration explicitly applies global factors and
time reflection, preserving the order of operations on contributions. These
private traversal routines handle edge-ended intervals with resolved internal
vertices; self-energy endpoint completion retains its separate insertion point.
No traversal direction, reflection policy or CPU workspace belongs in the shared
Backbone or BackboneRange. This leaves a future GPU executor free to choose its
own schedule, targeting high-end hardware such as GH200.

Host range discovery supports selected-vertex intervals beyond the two complete
correlator halves. It rejects unresolved external endpoints, which require their
own completion discovery. It retains the existing initial-sector mask policy;
this refactor does not change numerical pruning or support invalidation rules.

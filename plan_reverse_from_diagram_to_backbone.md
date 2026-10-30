# Self-contained backbones and block-sector paths

Drafted 2026-10-30 on `dev_backbone`, starting at `522ad52` (including the user's
header consolidation). Implementation evidence belongs in
`work_log_reverse_from_diagram_to_backbone.md`. Keep source and documentation
commits separate, with terse house-style messages; do not push.

## Objective and preparation flow

Restore Backbone as the complete diagram-specific description consumed by dense
execution. Give sparse execution one self-contained BackbonePath for either
endpoint operation. Keep the numerical input in const Action, the current
propagator, and separately prepared observable matrices.

```
Action metadata + order + kind + selected choices -> enriched Arcs
Topology + enriched Arcs                         -> Backbone
Backbone + sparse support/current G metadata      -> BackbonePath
Action + G + Backbone/BackbonePath + observables   -> numerical evaluation
```

One selection is produced at a time; this flow does not require materializing all
assignments or paths on the CPU. Flat indexing remains a compatibility adapter,
not a scheduling requirement. Future GPU optimization targets high-end hardware
such as GH200, not specifically the local RTX A4000.

## Shared representation

- Arc keeps direction, retained_flavour and pole_index and gains operator
  statistics and the selected pole's three-way sign (-1, 0, +1). A single metadata
  enrichment routine validates choices and obtains these fields from Action's
  statistics/poles. Preserve the existing flat-digit ordering. Retain a lightweight
  metadata overload for pure diagram fixtures, including first order with empty
  internal flavour/pole spaces. Reject nonfinite selected poles. Enriched internal
  arcs require legal statistics/direction, nonnegative indices and pole_sign in
  {-1, 0, +1}; the special arc never indexes numerical tables. Retain the existing
  physical precondition that each barred contraction couples operators of the
  same parity as its retained flavour; mixed-parity contractions are out of scope.
- Kind is an explicit input to arc preparation only. The special arc has no fixed
  flavour or pole: forward/backward means an interaction endpoint sum; external
  means supplied observables. That endpoint direction carries the distinction
  downstream; no redundant kind field is required in Backbone.
- `prepare_backbone(Topology const &, Arcs const &)` needs no Action or numerical
  tables. A vertex owns flavour, pole_index, barred/adjoint flags and kernel_sign.
  Unbarred internal vertices also need pole_index for their scalar kernel.
- Edge factors own `{pole_index, argument_sign}`. Whole-backbone normalization
  owns ordered `{pole_index, argument_sign, exponent}` terms. Preserve repetition
  and arithmetic order; do not deduplicate equal poles. Numerical TauKernel
  values remain in Action, and normalization is applied once per contribution.
- Backbone owns vertices, edges, the normalization recipe, internal direction
  sign and one endpoint descriptor: special_vertex, special direction and the two
  permutation signs. Order and before_special/after_special ranges derive from
  these fields. Endpoint reflection and global/permutation signs remain distinct.
- Internal statistics determine both permutation-sign cases during preparation.
  Endpoint statistics select a case during summation or observable completion;
  no endpoint-pair enumeration is required to prepare signs.
- Self-energy has 2m-1 edges; correlator has 2m. First-order correlator ranges each
  contain one propagator; a self-energy tail can be truly empty. Keep the existing
  range validation and explicit wrong-operation rejection.
- Remove runtime lookup through arc_idx. The enrichment factory overwrites derived statistics/pole signs; Backbone
  preparation checks local metadata invariants before use. Topology/Arcs remain construction inputs
  and need not outlive Backbone. Reprepare after changing choices/statistics/pole
  signs; no automatic invalidation or object-identity machinery.
- Remove the encompassing Diagram type once consumers use Backbone. Preserve
  the low-level topology helpers and their independent tests. A transitional
  compatibility wrapper is acceptable only inside an intermediate checkpoint.

## Separate F-bar construction from diagram instructions

Move OperatorReference, WeightedOperatorTerm and BarredOperatorContraction from
`diagram/arc.hpp` to the numerical Action area (`action.hpp`, root namespace).
Rename ArcOrientation to OperatorPairOrientation there: it describes adjoints,
coefficient-axis order and the reflected minus used by both F-bar construction
and numerical endpoint summation. Preserve its algebra and complex-coefficient
reference tests. No F-bar construction implementation should depend on diagram
headers. Diagram preparation needs only direction flags, not contraction helpers.

Replace SelfEnergyEndpointSum with the unified Backbone endpoint descriptor.
At execution, construct OperatorPairOrientation from that descriptor and obtain
flavour count from Action. Remove the obsolete `diagram/arc.hpp` after migration.
Dense operator storage exposes ordinary/barred numerical access by flavour/pole;
its interface should no longer require Vertex or Arc types. Sparse numerical
selection similarly consumes resolved path identifiers rather than a Diagram.

## One sparse BackbonePath

Use one owning value for complete contributions and selected backbone ranges:

- An owned shared Backbone recipe (initially copied as a whole for simplicity),
  a BackboneRange, and owned sector/dimension arrays for that range.
- Owned endpoint selections: ordered index lists at zero and at special. For an
  interaction endpoint they contain global Action flavour IDs; for external
  endpoints they index the supplied observable lists. The endpoint descriptor
  makes the interpretation explicit. Empty lists mean no completed endpoint
  operation for a segment; they must not imply an identity contribution.
- Output sector for self-energy is the starting sector, derived from the path.
  There is no separate SelfEnergyPath/CorrelatorPath class or required external
  SelfEnergyDiagram/CorrelatorDiagram wrapper.

Self-energy discovery owns one reusable workspace per Backbone call, matching the
segment-discovery visitor. Callers need no uninitialized path workspace; retained
results are copied explicitly. This initially gives up amortizing path-array
allocation across different arc assignments, a bounded implementation tradeoff.

A complete self-energy path contains the current full walk and endpoint candidates
filtered by exact support, preserving symmetry-group and within-group order.
Resolve group IDs to ordered flavour lists during discovery rather than storing
another executor-specific group-selection wrapper. Preserve nonclosing-path
rejection for nonzero endpoint coefficients and acceptance of exactly inactive
couplings. No magnitude cutoff is introduced.

A correlator still benefits from evaluating each segment once. Discover generic
segment paths and cache their numerical results in CPU-local tables. Completion
joins the two paths into another BackbonePath: slots 1..s come from before_special,
slots s+1..2m from after_special, and slot zero equals slot 2m. Its at-special list
selects the current tau observable; its at-zero list contains the allowed closing
observables in the current order. A reusable owning completion workspace avoids
repeated construction of the full backbone metadata.

Cache indices `(before_result, after_result)` may accompany the visitor callback
as CPU scheduling information. They must not be the meaning of the path itself.
A retained complete path must remain executable from Action/G/observables after
construction inputs, discovery workspace and path/cache tables are destroyed.
Observable matrices and CorrelatorStatistics remain numerical call inputs; the
latter selects the prepared permutation sign.
Keep a direct path evaluation entry point for this contract; the CPU orchestration
may use a private completion helper with cached segment results to retain reuse.

Range traversal does not apply whole-backbone normalization. Cache before-segment
results after applying the normalization and direction factors exactly once;
completion must not multiply them again. Standalone full-path execution applies
the same factors to its local before result. Preserve the existing external
permutation-sign application after accumulation. A generic path does not imply one generic
traversal algorithm: keep clear self-energy, forward-segment, reflected-segment
and observable-completion routines.

The full-Backbone copy per retained path is a deliberate initial metadata tradeoff.
Do not copy numerical matrices or make shared-pointer ownership a requirement.
Discovery remains streamed; existing single-target operator restrictions remain
explicit. No general branching executor is part of this change.

Ownership does not make a recipe model-independent: reuse requires compatible
Action indices, sector/basis layout, operator support, statistics and pole signs.
Changing G values with the same layout permits individual-path reuse. A whole
previously discovered path set can become incomplete when G's activity mask
changes and must then be rediscovered. Cross-Action validation is out of scope.

## Evaluation and binding

Dense core methods take Backbone, G and any observables, using evaluator-owned
Action. Sparse core methods take BackbonePath and numerical inputs, without an
additional Diagram/Backbone argument or an external path table for semantics.
Full-contribution entry points require path.range == path.backbone.range();
selected-range traversal is separate. Check endpoint cardinality before selecting
a tau observable. Empty endpoint lists produce no completed contribution.

Read operator and propagator blocks directly through the resolved identifiers.
Replace the mandatory `bind::*` stage and its numerical-view containers where
straightforward direct access now suffices. Small private lookup helpers are
appropriate; do not introduce a new operand hierarchy. Borrow observables by const
view/reference throughout the native execution chain. Persisted paths own logical
metadata only, and numerical owners must remain alive and unchanged during a call.

Keep Python-facing signatures, current result/accumulation behavior, numerical
conventions, paired-range ownership and manual reference algorithms unchanged.
The broader API-hardening and accumulation redesign from review_separation.md is
not part of this task. No new data layout, GPU kernels, MPI scheme or benchmarks.

## Reviewable implementation checkpoints

1. **Operator conventions and enriched arc input** (`[action] separate operator conventions`).
   Move construction helpers; add one validated arc-enrichment route and Action
   conveniences, with focused metadata tests. Keep existing evaluators usable.
2. **Self-contained Backbone execution** (`[diag] evaluate prepared backbones`).
   Resolve indices, kernels, endpoint geometry and signs into Backbone. Migrate
   dense execution, kernel helpers, sparse support/discovery inputs and remaining
   construction adapters. Update existing fixtures without changing their literal
   expected results. Remove Diagram once all consumers have migrated. Sparse path
   and binding wrappers may remain temporarily until the next checkpoints. If
   needed, split this into building shared/dense and sparse-removal sub-checkpoints.
   Adapt and compile backbone_cpu at every checkpoint affecting its interfaces.
3. **Generic owning sparse paths** (`[bs] prepare self-contained backbone paths`).
   Make segment/full paths own their instructions and endpoint choices. Replace
   specialized completion records with generic complete paths; retain CPU cache
   indices only as scheduling arguments. Validate ownership, path ordering and
   complete correlator slot composition before replacing numerical execution.
   Migrate the temporary bind functions to the endpoint lists in this checkpoint
   so removal of symmetry-set wrappers still leaves a building commit.
4. **Direct sparse path execution** (`[bs] evaluate backbone paths directly`).
   Consume the path alone with numerical inputs, remove obsolete binding/operand
   layers, preserve segment reuse, and adapt the benchmark consumer for compilation.
   Add standalone retained-path comparisons for both endpoint operations, including
   cached-versus-standalone correlator equivalence.
5. **Adversarial review and completion**.
   Audit stale references, sign/normalization placement, empty ranges, endpoint
   ordering, asynchronous-lifetime claims and accidental CPU schedule requirements.
   Fix concrete findings, run final acceptance checks, record remaining tradeoffs.

Where dependencies force two checkpoints together, document why; do not commit a
non-building intermediate interface. Follow every source checkpoint with a separate
work-log commit describing motivation, changed call flow, tests and remaining work.

## Validation and acceptance

- Preserve hand-derived flat-index, parity and kernel fixtures; numeric/manual
  comparisons and tolerances remain unchanged. Namespace/member access changes
  are expected; replacing independent oracles with the implementation is not.
- Cover three pole signs and both directions, repeated poles/factor order, mixed
  statistics, distinguished endpoint at first/final positions, empty self-energy
  tail, correlator trailing edge, reflection and paired/unpaired equivalence.
- Add proof that prepared Backbone works after topology/arcs are destroyed, and
  that retained paths of either operation work after discovery/cache metadata is
  destroyed. Verify mutation of a copied preparation does not affect its source.
- Preserve sparse rectangular sector dimensions, zero masks, exact operator/bar
  support, nonclosing-path behavior, and endpoint/global flavour ordering.
- Verify that several observable joins can reuse a segment without repeating
  normalization; compare cached completion against standalone full-path execution.
- Include first-order empty-internal-choice preparation without constructing an
  Action and malformed arc metadata rejection before lookup.
- Reconfigure CPU cbuild when files move, use ~/env/xca-cpu.sh, build at -j6,
  run relevant focused tests per checkpoint. Final acceptance: all CPU/Python
  CTest entries with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and parallelism 2;
  compile backbone_cpu; inspect regenerated bindings and git diff --check.
- No new benchmark runs or GPU claims. Preserve the user's header organization,
  local scripts/caches, modified .gitignore and all unrelated untracked files.

## Plan review

Three independent adversarial reviews covered shared preparation, sparse execution
and the numerical/F-bar boundary. No architectural blocker was found. Revisions:

- State metadata validation, factory overwrite behavior and the existing same-parity
  contraction precondition; external endpoints must be rejected before numerical
  interaction-orientation construction.
- Distinguish owned recipe lifetime from Action/layout compatibility and propagate
  the activity-mask rediscovery rule. Keep CorrelatorStatistics explicit.
- Define exactly-once cached normalization and require reused-segment and
  standalone-path comparisons.
- Migrate temporary binding consumers with endpoint record removal, and keep the
  benchmark compiling at each interface-changing checkpoint. Allow a building
  subdivision of the broad Backbone migration if it improves reviewability.

Reviewed and revised 2026-10-30; ready for implementation.

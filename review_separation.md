**Review of diagram planning, evaluation and input-data separation**

2026-10-26. Reviewed `c++/triqs_xca` on `dev_backbone`, source revision
`012ca26`. Three agents independently reviewed the shared diagram layer,
evaluator/planning/binding code, and numerical data/construction code. The primary
review reconciled their findings, checked cited call paths and tests, and examined
the public input boundaries. All three reviewers then cross-checked the draft;
their corrections on support reuse, input provenance and lifetimes are included. This is a static architecture review: no source
changes, builds, test runs or benchmarks were performed. The preceding work log
records 76 passing CPU/Python CTest entries; that is prior evidence, not a new
validation result or proof that every input contract is enforced.

The recent refactoring has established useful boundaries. The next improvement
should be to make those boundaries reliable and easy to use. Most recommendations
below can be implemented with ordinary functions, const references and existing
containers. Another layer of diagram classes, a second numerical model, or a
common evaluator inheritance hierarchy would add complexity without addressing
the main issues.

The intended ownership is already mostly recognizable:

| Object or layer | Meaning and lifetime | Boundary to preserve or clarify |
| --- | --- | --- |
| `Action<OperatorStorage>` | Persistent numerical input, including scalar tables and F-bar storage | One coherent owner; no retained `atom_diag` dependency |
| `Topology`, `Arcs`, `Backbone`, `Diagram` | Connectivity, choices and scalar placement recipe for one assignment | Logical description; compatible input is required to prepare and execute it |
| Sparse support and paths | Model connectivity, then selected sector walks | Separate immutable support from the current propagator's activity mask |
| `bind::*` and operand structs | Host numerical views for a selected path/completion | Borrow numerical arrays; own retained path metadata |
| Evaluator | Traversal, scratch storage, contractions and accumulation | Separate native numerical operation from public input/output adaptation |

Actual arc indices, explicit endpoint ranges and the compact endpoint sum should
stay. So should the separation between kernel recipes in Backbone and numerical
TauKernel tables in Action. Independent indexing/sign fixtures and the manual reference routines are valuable:
do not replace their expected values or independent algorithms with the same
high-level helpers being tested.

**1. Make input compatibility a checked boundary. Highest priority.**

Several public-looking entry points accept objects whose validity is stronger
than their type expresses.

The TRIQS propagator adapters check block counts and array shapes, but do not
compare the input meshes with `action.tau_mesh`. A propagator with the same number
of nodes but a different beta/grid can therefore be interpreted using the
Action's convolution operators. Check compatible beta, nodes and mesh semantics
once at the public GF boundary. Raw array/BlockDiagOpFun entry points have no mesh
to inspect; explicitly document their same-grid and same-basis precondition and
check the shapes they do carry. Evidence: [dense input conversion](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.cpp:81),
[sparse input conversion](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.cpp:479).

Likewise, a prepared Diagram is not independent of all numerical input.
Preparation checks its indices against supplied dimensions and chooses stable
vertex/edge kernels using the selected pole signs. Execution accepts that value
against an evaluator's Action without checking this compatibility. Foreign
flavour/pole indices can be out of range; changing the pole signs can defeat the
stable factorization. The latter is a numerical stability risk, not a claim that
every sign change invalidates the algebraic identity. Public mutation of the
Diagram's source fields can also leave its derived backbone stale. Evidence:
[preparation contract](/home/hugstr/dev/xca/c++/triqs_xca/diagram/diagram.hpp:13),
[pole-sign choice](/home/hugstr/dev/xca/c++/triqs_xca/diagram/backbone.cpp:32),
[prepared dense execution](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.cpp:316).

Keep plain owning values and the documented re-preparation rule. Add an easy
preparation entry point using the evaluator's Action, and a deliberate validation
boundary for externally prepared values. Trusted inner loops should not rebuild
or repeatedly validate every recipe. Bounds/consistency checks cannot infer the
meaning or ordering of operators and poles; common ordering remains an explicit
caller contract. No model identity registry, automatic cache invalidation, or
copied numerical data is needed.

There is also a concrete malformed-input problem in operator construction:
`BlockOpSymQuartet` accesses `Fs[0]`/`F_dags[0]` before rejecting empty groups and
uses negative symmetry labels as array indices before validating the labels.
Check counts, label bounds and compatible group shapes before indexing or
constructing bars. Dense `FSet` should similarly check unbarred matrix shapes at
its own construction boundary, rather than relying on a later Action factory.
These are source-level findings; no failing runtime case was executed in this
review. Sparse construction also rejects nonfinite coefficients, whereas the
shared Action and dense FSet checks do not; make this input rule consistent at
shared scalar validation if it is retained. Evidence: [coefficient check](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/block_op.cpp:470),
[quartet construction](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/block_op.cpp:496),
[label indexing](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/block_op.cpp:517),
[FSet construction](/home/hugstr/dev/xca/c++/triqs_xca/dense/fset.cpp:6).

Suggested checks: same-shaped G on an incompatible mesh; foreign prepared indices;
pole-sign changes under the chosen validation rule; empty/negative-label sparse
inputs; and malformed dense operator shapes. Existing wrong-kind and recovery
tests should remain. The existing Action DLR test checks a legacy constructor's
grid reconstruction, which does not cover runtime G compatibility.

**2. Give the native evaluator API one clear accumulation contract. High priority.**

Dense prepared self-energy evaluation adds into a public `Sigma`; callers manage
resetting it. Sparse prepared `compute_self_energy` initializes its accumulator,
evaluates, constructs a TRIQS block GF and resets. Both are usable, but there is
no equivalent numerical boundary that a caller can use consistently across
representations. Evidence: [dense prepared API](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.hpp:115),
[sparse prepared computation](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.cpp:300).

Prefer a native operation whose name explicitly means accumulation, with a
caller-supplied output in the representation's natural storage. Convenience
`compute_*` methods can allocate/clear, call that operation and wrap the result.
This is a shared semantic contract, not a requirement for identical dense/sparse
argument types or a virtual base class. Correlator methods should state the same
add-versus-return distinction. Document exception guarantees and that one
instance's scratch storage cannot serve concurrent evaluations.

As a small first step, group compatibility overloads separately from the
canonical prepared API in the headers, and remove mutable duplicate metadata
from normal caller control. Both evaluators expose writable copies of beta,
rank and flavour counts even though the numerical model is a const Action.
Changing those copies can disagree with the tables and allocated scratch arrays.
Read-only compatibility properties/accessors can preserve Python reads. A private
workspace member can group scratch arrays if helpful; it need not become another
public framework. Evidence: [dense state](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.hpp:39),
[sparse state](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.hpp:88).

Validate with two explicit additions versus a summed call, independent
`compute_*` calls, empty selections, and recovery after rejected input. Preserve
the existing copy/lifetime tests and public Python compatibility during migration.

**3. Make Action construction coherent through all supported routes. High priority.**

The named dense/sparse `make_action` factories rebuild bars from the Action's
owned coefficients. This is the right guarantee. The generic public Action
callback constructor, however, checks scalar metadata but cannot ensure that the
callback returned operators matching those coefficients, statistics and block
dimensions. Standalone Action fields remain mutable before being moved into a
const evaluator, so coefficients can also be changed without updating derived
tables. Direct evaluator constructors trust the supplied Action. This is an
unchecked alternative construction contract, not a defect demonstrated in the
normal factories. Evidence: [Action construction](/home/hugstr/dev/xca/c++/triqs_xca/action.hpp:70),
[dense factory](/home/hugstr/dev/xca/c++/triqs_xca/dense/action.hpp:8),
[sparse factory](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/action.hpp:10).

Make named factories the documented supported construction path. Treat the raw
callback as an internal construction mechanism, and expose built Actions for
read-only use. Shape checks alone cannot establish that externally supplied bars
match coefficients, so retaining factory reconstruction is preferable to trying
to validate arbitrary callbacks after the fact.

There is a smaller simplification available at the same boundary. A numerical
caller currently supplies an `FSet` or quartet whose bars may already have been
computed, only to have them rebuilt by `make_action`. Accept unbarred arrays or
groups directly as another factory input, then construct bars once inside Action.
Reuse existing groups rather than inventing an alternative persistent operator
representation; [BlockOpSymSets](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/atom_diag.hpp:20)
already provides much of the sparse input shape. Keep the existing overloads as
adapters.

Retain `block_dims`: sectors in which all operators vanish still have a Hilbert
space dimension. Operator support cannot recover that information. Sparse const
accessors already return const views; this review did not find a mutable-view
leak through a const Action. The lifetime and stale-bar tests in
[action.cpp](/home/hugstr/dev/xca/test/c++/action.cpp:8) should protect these decisions.

**4. Let sparse planning consume metadata without prescribing a CPU schedule. High priority for the next backend.**

Path discovery is already outside the arithmetic kernels, but its functions
still accept full numerical propagator/operator containers. They largely read
sector counts, dimensions, transitions, symmetry-group maps and the propagator's
current zero mask. The persistent `OperatorSupportTable` is private to the
evaluator even though it describes the immutable operator input. Evidence:
[backbone walk](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/backbone_paths.cpp:5),
[self-energy walk](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/self_energy_paths.cpp:7),
[support ownership](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.hpp:35).

Pass existing support, dimensions and group metadata through narrow const
arguments/views. Keep the current propagator activity mask separate because it
changes when G changes. Construct persistent support alongside the immutable
sparse input, or reuse the existing independently constructible support table
across execution contexts. It is derived metadata, not a second matrix
representation. The legacy unknown dimension
`-1` needs a deliberate adapter that resolves available dimensions from G before
traversal.

`bind::self_energy` currently receives operators, two coefficient-value tables
and statistics independently. At its entry point, accepting the corresponding
Action would make their common provenance clear; low-level slice helpers can
retain narrow arguments. Keep numerical binding in `block_sparse::bind`, outside
the shared diagram namespace. Evidence:
[self-energy binding](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/bind/self_energy.cpp:16).

The sparse correlator currently computes and retains both sets of segment
results, then joins them. Its `CorrelatorDiagram` refers to these temporary path
and result tables. That is a valid CPU scheduling choice, but should not become
the required input format for another executor. Keep the semantic operation
“connect these backbone endpoints with these observable operators” distinct from
whether a backend caches, streams, batches or recomputes segment results.
The current support container can describe multiple targets per source, but the
CPU walker calls `single_target`, and correlator lookup assumes one segment per
starting sector. General branching is not implemented by having a more general
metadata format. Keep this executor restriction explicit when exposing planning
APIs. Evidence: [single-target check](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/operator_support.cpp:32),
[correlator lookup](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/correlator_diagrams.cpp:7).

Evidence: [correlator execution](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.cpp:421),
[completion table indices](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram/correlator.hpp:5).

The current diagram API does **not** force the CPU to enumerate and transfer
every contribution. Preserve that freedom. Metadata-only planning inputs are
useful for independent testing and eventual packing; they do not require
materializing a complete host plan. Test metadata-based visitors against the
existing paths under changing G masks, zero operators and rectangular sectors.

**5. Prepare observables once and borrow them throughout evaluation. Medium priority.**

Both evaluators still perform symbolic parity classification, atom_diag matrix
conversion and basis rotation inside the `compute_one_time_correlator` adapter.
The core already accepts numerical operators, so extract these conversions into
simple representation-specific preparation helpers. The existing API can call
the helper; a C++ caller can reuse its result across topologies and index ranges.
Do not put requested observables into Action or retain atom_diag there.
Evidence: [dense observable setup](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.cpp:460),
[sparse observable setup](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.cpp:620).

Dimension checks do not establish a common basis: two atom_diag instances may
have equal dimensions but different state/sector ordering. Document that prepared
observables and G use the Action operators' occupation basis and ordering.
Retaining atom_diag merely to compare object identity would not solve this.

Core correlator methods take owning arrays or `vector<BlockOp>` by value, and
selection loops pass the same lvalues for each index. These signatures induce
repeated copies of unchanged operator data. Use const array views and const
spans/references in the internal call chain, with one owner at the adapter/caller.
Evidence: [dense loop](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.cpp:500),
[sparse loop](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.cpp:599).
The cost has not been measured here.

Do not remove every copy indiscriminately. Operand structs intentionally own path
metadata while borrowing numerical data, so retained operands survive reuse of a
visitor's workspace. That is tested and useful. Preserve this lifetime contract,
especially if later execution becomes asynchronous: owned path metadata does not
extend the lifetime of borrowed Action/G/observable arrays. Their owners must
remain alive and unchanged until execution completes. Evidence:
[operand ownership](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/backbone_operands.hpp:7),
[retained-operand tests](/home/hugstr/dev/xca/test/c++/block_sparse/prepared_evaluator.cpp:80).

**6. Finish results at the call boundary, and distinguish absence from pruning. Medium priority.**

The sparse selected self-energy loop calls a helper that scans all accumulated
Sigma blocks for exact zeros after every flat index. Path discovery subsequently
reads G's mask, not Sigma's mask. The whole-topology helper already performs this
scan once at the end. Move final output-mask classification/conversion to the
common completion boundary, retaining conservative active flags while adding.
Evidence: [per-index finalization](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.cpp:173),
[selected loop](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/diagram_evaluator.cpp:266),
[exact-zero scan](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/block_op.cpp:95).

The mask is not always an exact statement that a block is nonzero: setters and
addition mark a block active without inspecting every entry. Its useful contract
is “absent implies zero; active may be zero until classification.” Document this,
and consider an `is_active` accessor instead of scattering the `-1`/`0` encoding.
Test exact cancellation, empty output, and subsequent accumulation. No speedup
estimate follows from source inspection alone.

The historical magnitude cutoff is absent on this branch. Likewise,
`select_diagonal_self_energy` examines already-bound input coefficients to reject
a nonclosing, nonzero contribution; it does not synchronize on a newly evaluated
result. Keep this structural/output-partition rule explicit and separate from
any future optional numerical pruning. Evidence:
[partition check](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/self_energy_operands.cpp:10).

**7. Clarify selection and shared sign semantics with small helpers. Medium priority.**

Paired dense evaluation deliberately treats a range differently from an explicit
vector: ranges process only owning direction-zero seeds, whereas vectors/scalars
evaluate both partners for every supplied seed. Converting the full range into a
vector can therefore double-count pairs. This is documented historical behavior,
not a new regression. Give the internal adapters explicit pair-seed versus
assignment-range meanings; do not infer the mathematical operation from
`FlatIndexSelection::is_range()`. Preserve existing Python semantics until an
intentional API migration. Evidence:
[pair ownership](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.cpp:145),
[partition tests](/home/hugstr/dev/xca/test/c++/flat_index_range.cpp:224).

The global self-energy order/orientation sign is repeated in dense evaluation
and sparse binding. Put that specific formula in one named diagram helper so
binding only resolves numerical operands. Keep permutation signs, endpoint
reflection signs and numerical kernel normalization distinct: they enter at
different stages. Evidence: [dense sign](/home/hugstr/dev/xca/c++/triqs_xca/dense/diagram_evaluator.cpp:327),
[sparse sign](/home/hugstr/dev/xca/c++/triqs_xca/block_sparse/bind/self_energy.cpp:51).

**8. Improve file and name boundaries after the contracts. Lower priority.**

| Opportunity | Small change and constraint |
| --- | --- |
| Heavy/transitive diagram includes | `diagram/tau_kernel.hpp` includes topology and arcs for two small recipe types; `dense/fset.hpp` includes the whole Diagram. Use direct dependencies/declarations. |
| Statistics pulls Action into topology | The placement in `action.hpp` was deliberate. First prune other includes; an opaque enum declaration with sign implementation in `.cpp` is an option. Do not move the enum back solely to obtain a pure dependency diagram. |
| Partial preparation entry point | Declare/document `prepare_vertices` with backbone preparation. Make clear that it is a helper, not a complete kernel-prepared backbone. Keep its independent tests. |
| Numerical helper and axis names | `update_hybridization`, `hyb_coeffs`, and comments describing every flavour as a fermion now also cover retarded operators; `BlockOpSymSetBar::get_num_time_nodes()` returns a pole extent. Use accurate names at touched internal boundaries; retain physical-family qualifiers where they distinguish inputs. |
| Block storage module | `block_op.*` groups propagator storage, plain operators, symmetry groups and barred contractions. If split, use these concrete responsibilities and move definitions without changing algorithms in the same commit. Avoid introducing one new abstraction per storage rank. |

The duplicate topology-only correlator-range helper is not itself a design
failure: callers should be able to ask geometry questions without supplying
numerical poles. Small formula consolidation is optional.

A practical sequence would be:

| Reviewable change | Intended result | Validation focus |
| --- | --- | --- |
| Harden constructors, then evaluation input checks in a separate checkpoint | Reject malformed shapes/labels before indexing; establish mesh and prepared-diagram contracts | Focused rejection tests, then existing evaluator tests |
| Clarify native accumulation and compatibility adapters | One understandable lifecycle for add/compute/reset; read-only derived metadata | Repeated calls, explicit additions, exceptions, Python API |
| Extract observable preparation and pass const inputs | Atom-diag conversion stays outside the numerical loop; reusable numerical observables | Existing correlator/changed-G tests, lifetime checks |
| Narrow sparse planning inputs and reuse support metadata | Planning depends on connectivity/dimensions/masks, not matrix-container APIs | Compare path sequences, rectangular/zero sectors, mask changes |
| Move finalization and share the global sign formula | Less work and less domain algebra inside adapters/binding | Cancellation, mixed statistics, paired/plain comparisons |
| Clean names, headers and source layout | Easier navigation without new architectural machinery | Build and relevant existing tests |

Keep each source change separate from its explanatory work-log commit. The first
change should not simultaneously redesign all constructors, outputs and sparse
storage. Implement one boundary improvement, explain its before/after call flow,
and preserve a functioning evaluator at each checkpoint.

Further performance decisions need measurements. After removing obvious owning
argument copies, measure observable setup, path discovery, binding, contractions,
accumulation/finalization, allocations and peak cached segment storage separately.
Include many scalar blocks and heterogeneous rectangular blocks, and vary the
number of indexed evaluations. This would determine whether packed metadata or a
different correlator schedule earns its complexity. Future GPU experiments should
target high-end hardware such as GH200; the local RTX A4000 is not the optimization
goal. Nothing here calls for a new GPU experiment or a complete MPI/Cartesian-work
redesign now. If a backend needs compact work bounds, expose them directly rather
than expanding them through a host callback or a vector of every Diagram/path.

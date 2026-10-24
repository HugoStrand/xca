# Action ownership sketch

Draft for discussion, 2026-10-24, based on `dev_backbone` after the pole and
reflection convention fixes. This proposes an incremental refactoring, not a
new evaluator or authorization to implement it.

## Recommendation

Introduce `triqs_xca::Action` in `c++/triqs_xca/action.hpp` as the owner of the
persistent model description. Keep the existing numerical representation:
`ArcData`, the atom_diag instance, and second-quantized retarded operators.
Use **couplings** for the Action member/accessor holding the combined scalar
data: `action.couplings()`. Initially its type remains `ArcData`; changing its
name is not necessary to establish ownership.

Keep the existing dense `FSet` and block-sparse `BlockOpSymQuartet` outside
Action. They materialize representation-dependent operators and F-bars from
Action. The evaluator owns them and its working buffers. The propagator remains
an independent input, since it changes during self-consistency.

Action is immutable after construction. Replacing fitted data produces a new
Action and rebuilds the corresponding F-bar/evaluator state. There is no second
persistent set of split coefficients behind the combined coefficients.

## Combine the inputs exactly as the current code does

Both evaluators already use `hyb::get_extended_coefficients`. With `h` fundamental
fermion operators and `d` retarded operators, use one flavour space of size
`h + d`:

| Global flavour | Operator before taking its adjoint | Scalar coefficient block | Statistics |
| --- | --- | --- | --- |
| `0 <= a < h` | annihilation operator `c_a`, in atom_diag fundamental-operator order | hybridization | fermionic |
| `h <= a < h+d` | `dynint_ops[a-h]` | retarded interaction | bosonic |

For each shared physical pole `poles[p]`, construct the single stored tensor

```text
coeffs[p] = [ hyb_coeffs[p]       0          ]
            [      0       dynint_coeffs[p] ]
```

The off-diagonal blocks are exactly zero. Entries inside either diagonal block
may be off-diagonal in flavour: coupled retarded operators remain a full matrix,
not a list of independent scalar interactions. No operator matrices or symmetry
sets are needed to construct this tensor.

For example, with `c_up`, `c_down`, `n_up`, `n_down`, the combined labels are
`0, 1, 2, 3`. A retarded coefficient `D_up,down[p]` occupies `coeffs(p,2,3)` and
couples `n_up^dag` to `n_down` in the forward orientation. This does not create
hybridization/retarded cross terms. A diagram can nevertheless contain arcs of
both families, just as it does now.

The input remains **already fitted poles and coefficients**. The existing
Python fitting path fits hybridization and the retarded kernel together and
then splits their coefficient blocks. Action can accept those two blocks and
assemble them once. There is one common pole list; no independent-pole merging,
refitting, extra pole offsets or new fitting API is needed. Multiple physically
separate retarded kernels can already be expressed by concatenating their
operator lists and placing their coefficients in diagonal subblocks of the
retarded block, on that same pole list.

The retarded coefficients retain their current normalization. In particular,
Action does not insert a factor of one half into the retarded action term.
Bosonic operator statistics do not require a different numerical kernel here:
the fitted scalar functions already use the common cppdlr representation.
Action does not reinterpret their coefficients based on statistics.

Construct `ArcData` once from this tensor, retaining its existing derived tables:

- `poles`: physical frequencies, unchanged from input;
- `coeffs`: the combined tensor above;
- `values`: combined scalar functions at the imaginary-time nodes;
- `values_reflect`: unsigned `C(beta-tau)`;
- `kernel_values`: the existing tables evaluated with dimensionless `beta*poles`.

These tables are the existing precomputations, not another model representation.
If callers need the original hybridization or retarded coefficient block, expose
const slices of the combined tensor. Do not store the original blocks as well.
The construction-time concatenation order makes these slices straightforward.

## Ownership and a concrete interface

Illustrative declarations, not a proposed inheritance hierarchy:

```cpp
template <bool IsComplex>
class Action {
  triqs::atom_diag::atom_diag<IsComplex> atom_diag_;
  std::vector<triqs::operators::many_body_operator_complex> dynint_ops_;
  ArcData couplings_;
  diagram::ArcCatalog catalog_;

public:
  // Inputs: atom_diag, mesh, physical poles, hybridization coefficients;
  // optionally retarded operator expressions and retarded coefficients.
  // Construct couplings_ and catalog_ from these inputs once.

  auto const &atom_diag() const { return atom_diag_; }
  auto const &dynint_ops() const { return dynint_ops_; }
  ArcData const &couplings() const { return couplings_; }
  diagram::ArcCatalog const &catalog() const { return catalog_; }
};
```

The boolean follows the existing atom_diag real/complex template parameter.
Evaluator constructors can remain templated on it, as they are today; the
numerical evaluator classes need not become templates. They can borrow the
non-templated scalar data and catalogue after constructing their F-bar objects.
A correlator setup operation that needs atom_diag receives the Action and uses
`action.atom_diag()`, rather than accepting an unrelated atomic realization.

Action owns its atom_diag, preferably moved at construction where possible.
The target solver structure should refer to this owned instance rather than
keep another persistent copy. Creation/annihilation operators are identified
by its fundamental-operator ordering; only retarded operator expressions need
an additional stored list. Their matrix representations remain in F-bar setup.

A prospective call pattern is:

```cpp
Action<true> action(ad, mesh, poles, hyb_coeffs, dynint_ops, dynint_coeffs);
dense::DiagramEvaluator evaluator(action); // builds FSet, borrows scalar data
// Alternatively: block_sparse::DiagramEvaluator evaluator(action);
auto sigma = evaluator.compute_self_energy(G, topology_arr);
```

For this borrowing interface, **Action must outlive the evaluator**. That is
stronger than the minimum guarantee that it lives during an individual call,
but avoids repeatedly passing model data that must match cached F-bars. In the
solver, construct/own Action before the evaluator and retain it for the same
period. Python bindings must retain its owner, not just a borrowed C++ pointer.
This lifetime policy is a recommendation for review, not an existing guarantee.
No mutable alias may modify the owned model through a const accessor.

Mesh and beta are obtained from `couplings().tau_mesh`; avoid adding independent
copies to Action. Mutable convolution workspaces remain evaluator state.

## ArcCatalog and the remaining binding work

`ArcCatalog` is metadata for the combined flavour space. It contains family
membership, statistics and the ordered external hybridization flavour list.
It does not contain poles, coefficients, operators or a chosen topology.
Action owns one catalogue, built by the existing
`diagram::hybridization_retarded_catalog(h,d)` adapter. Evaluators stop owning
separate copies. Keep the metadata type's existing namespace initially; its
ownership by Action does not require another namespace refactoring.

`diagram::Arcs` has a different role: it records chosen directions, pole indices
and retained flavours for one diagram. Those indices refer to Action's common
flavour and pole spaces. `diagram::OperatorReference{flavour, adjoint}` and
`BarredOperatorContraction` already express the required operator choices and
partial contractions; they need no Action-specific counterparts.

There is also a separate block-sparse partition into operator symmetry sets.
It belongs to F-bar construction and must not be confused with catalogue
families. A bosonic family may contain several such symmetry sets.

Action simplifies access to persistent inputs, but does not eliminate all bind
functions:

| Existing work | Treatment |
| --- | --- |
| Select tau-kernel table views for a diagram | Keep `bind::tau_kernels`; source tables from Action |
| Look up ordinary/barred operator matrices | Keep representation-specific F-bar access |
| Select propagator blocks and form backbone operands | Keep block-sparse binding; propagator is separate |
| Bind endpoint sums and correlator joins | Keep where they assemble useful operands; take scalar data/statistics from Action |
| Assemble persistent coefficient blocks and family metadata | Perform once at Action construction |

Bound operands are temporary views for CPU execution, not an obligatory second
persistent model. A future GPU evaluator may index Action data directly or
upload its numerical arrays once and use another execution strategy. Neither
Action nor its catalogue requires CPU enumeration or transfer of every backbone
path. GPU packing and work partitioning remain separate future decisions;
performance work targets high-end hardware such as GH200, not the local RTX A4000.

## Scope and incremental implementation

Common construction should check coefficient shapes and shared pole-axis lengths,
and that hybridization/retarded flavour counts match atom_diag/the operator list. Keep the
existing bosonic-retarded-operator model assumption; generalizing statistics is
outside this ownership change.

Keep representation-specific restrictions in the corresponding F-bar builder:
dense retarded setup currently requires a single atom_diag subspace; block-sparse
setup requires representable operator connection maps and rejects coefficients
coupling incompatible symmetry sets. Action must not reject a model merely
because one evaluator cannot represent it.

A reviewable implementation sequence would be:

1. **Introduce Action ownership and common assembly.** Add the container using
   existing `ArcData` and catalogue construction. Test combined blocks, metadata,
   physical poles, derived tables, and the empty-retarded case. Evaluators still
   work through their existing constructors at this checkpoint.
2. **Let F-bar builders consume the combined data.** Adapt dense and block-sparse
   setup to use Action's operators and coefficient tensor. Remove their repeated
   calls to `get_extended_coefficients`; retain matrix layouts, basis rotations,
   barred contractions and symmetry grouping. Compare old/new operator results.
3. **Make evaluators and the Python solver use Action.** Borrow its scalar data
   and catalogue; retain F-bar and workspaces in each evaluator. Establish owner
   lifetimes and rebuild rules, and adjust correlator construction to use its
   atom_diag. Remove duplicated persistent data. Preserve low-level prebuilt
   operator test entry points without inventing a dummy atom_diag for them. Their
   fixtures can own `ArcData` and a catalogue and supply them, together with
   prebuilt F-bars, to the same numerical initialization used by the Action
   constructor. This needs no abstract model interface or second data format.

Each step gets a focused code commit and a separate work-log checkpoint. Step 3
can be split between C++ and Python if its diff becomes hard to review; any old
constructor retained during migration must have a documented ownership path,
not dangling references to a temporary Action.

Reuse the existing convention, interaction-contraction, mixed-family,
dense/block-sparse agreement, symmetry-set, correlator and Python solver tests.
Only add tests for new ownership/assembly contracts or uncovered behavior.
No new diagram partition API, GPU implementation, F-bar redesign or benchmark
campaign is part of this sketch.

## Source anchors

- `c++/triqs_xca/hyb.cpp`: `get_extended_coefficients` and scalar evaluation.
- `c++/triqs_xca/arc_data.{hpp,cpp}`: existing scalar storage/precomputations.
- `c++/triqs_xca/diagram/arc_catalog.{hpp,cpp}`: families and statistics.
- `c++/triqs_xca/diagram/arc.hpp`: orientations and barred contractions.
- `c++/triqs_xca/{dense,block_sparse}/dynint.cpp`: operator assembly.
- `c++/triqs_xca/{dense,block_sparse}/diagram_evaluator.cpp`: repeated ownership.
- `python/triqs_xca/solver.py`: joint fitting, normalization and evaluator setup.

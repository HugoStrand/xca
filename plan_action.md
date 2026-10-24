# Action refactoring plan

Revised and reviewed 2026-10-24 on `dev_backbone`. This version supersedes the
previous proposal to retain atom_diag in Action. Implementation authorized on 2026-10-24 as a persistent goal. Progress and
validation are recorded in [work_log_action.md](work_log_action.md).

## Agreed ownership

Action is the persistent numerical input for one evaluator representation,
including its operator/F-bar precomputations. Template it only on that existing
operator-storage type:

```cpp
template <typename OperatorStorage>
struct Action {
  triqs::mesh::dlr_imtime tau_mesh;
  TauKernelValues kernel;
  nda::vector<double> poles;
  nda::array<nda::dcomplex, 3> pole_coeffs;
  nda::array<nda::dcomplex, 3> tau_values;
  nda::array<nda::dcomplex, 3> tau_values_reflect;
  std::vector<diagram::Statistics> statistics;
  int n_hyb;
  nda::vector<long> block_dims;
  OperatorStorage operators;
};
```

This is a member sketch, not a prescription for public mutation or constructor
order. Construct consistent state, then expose it read-only. Counts other than
`n_hyb` are derived from the arrays/representation where possible.

The dense evaluator owns `Action<dense::FSet>`; the block-sparse evaluator owns
`Action<block_sparse::BlockOpSymQuartet>`. Action replaces ArcData, ArcCatalog,
and the separate evaluator F-bar member. There are no kind/family objects and
no separate persistent list of ordinary operator matrices beside `operators`.

FSet/BlockOpSymQuartet continue to own ordinary matrices, adjoints and barred
contractions in their existing packed layouts. They need no references back
into Action. Operator symmetry grouping and support information retain their
representation-specific roles. Mutable diagram workspaces and the propagator
remain outside Action; immutable support metadata can initially stay in the
evaluator, derived from `action.operators`.

**No atom_diag object or reference is retained.** It is an adapter input used to
construct matrices, not the local Hamiltonian needed by the numerical evaluator.
There is no atom_diag copy, lifetime extension, real/complex reference variant,
abstract base class or runtime backend selector. Dense and block-sparse Actions
are different numerical realizations of the model; this is deliberate.

## Construction paths and operator meaning

Keep the existing Python-facing evaluator constructors, including argument names,
real/complex atom_diag overloads and `dynint_ops`. Construct Action inside them.
The existing solver calls need not change:

```python
evaluator = DenseDiagramEvaluator(
    poles, hyb_coeffs, mesh, ad, dynint_ops, dynint_coeffs)
# The corresponding BlockSparseDiagramEvaluator calls also remain unchanged.
```

The atom_diag path constructs a temporary ordered symbolic list:

```text
h = ad.get_fops().size()
ops = [c_0, ..., c_(h-1), dynint_ops[0], ..., dynint_ops[d-1]]
```

Use atom_diag's fundamental-operator ordering. There is one operator per flavour;
adjoints are derived, not additional coefficient-axis entries. Resolve matrices
in the occupation-number basis used by the evaluator, undoing the atomic
eigenbasis rotation as current code does. Symbolic `ops`/`dynint_ops` are setup
inputs and need not persist in Action.

The direct numerical path accepts operator matrices and their layout/statistics,
without atom_diag. Existing C++ test entry points accepting FSet or
BlockOpSymQuartet can delegate to this construction. No dummy atomic model or
partially initialized Action is needed. Preserve tests that deliberately supply
alternative groupings and synthetic matrices.

If an ungrouped block-operator input is useful, use a small setup record with
`connection[b]` and `blocks[b]`: source sector b maps to the target sector in
`connection[b]`, with -1 for absence. Rows belong to the target sector and columns
to the source. Reuse/consolidate the existing BlockOp/op_block_mat patterns;
do not introduce another persistent operator container alongside F-bar storage.
This compact format retains the accepted single-target and injective-connection
restriction needed to represent adjoints. A dense matrix is the one-sector case.

The direct constructor must also receive:

- `block_dims`, including sectors on which all operators vanish; matrix shapes
  alone cannot recover these. Dense evaluation has one full-space dimension.
- `statistics` in global flavour order; arbitrary matrices alone do not determine
  fermion parity.
- `n_hyb`, identifying the first fundamental-operator flavours used by
  single-particle observables. Statistics cannot substitute for this information.

Keep legacy low-level overloads as thin adapters where useful. An older
quartet-only input can infer dimensions where blocks exist; preserve its current
unknown-sector handling explicitly until callers provide full dimensions.
Do not invent dimensions for missing sectors. Tests of custom layouts should
prefer the explicit-dimensions path.

Correlator methods retain their existing atom_diag argument and construct their
observable matrices separately in the same evaluation basis. Preserve current
compatibility checks and behavior. Their operators are not added to Action's
model operator list. A direct matrix correlator path already fits this boundary.

## Coefficients, statistics and consistency

Both construction paths use one physical pole list and one combined tensor:

```text
pole_coeffs[p] = [ hyb_coeffs[p]       0          ]
                [      0       dynint_coeffs[p] ]
```

An entry `(p,a,b)` couples operator a's adjoint to operator b in a forward arc.
Backward orientation, index reversal and reflection signs continue to come from
`diagram::ArcOrientation` and `BarredOperatorContraction`. There is no new factor
of one half or change of fitted scalar-kernel normalization. Coupled dynamical
operators remain a full matrix within the retarded block.

Inputs are already fitted poles/coefficients on the shared pole set. Keep the
Python joint-fit/split procedure. Assemble the combined tensor once, using the
current `hyb::get_extended_coefficients` pattern. Do not retain the split blocks
as additional evaluator storage. Setup can take const slices when needed.

Build F-bars from this same `pole_coeffs`. For prebuilt operator-storage inputs,
validate shape/layout and coefficient admissibility, then rebuild barred
contractions from their ordinary matrices/adjoints and the supplied coefficients.
Do not assume existing bars match merely because their dimensions do. Preserve
custom symmetry labels/grouping while rebuilding. Current quartet-only checks
cover shapes and cross-set coefficients, not numerical agreement of every bar.

Construct scalar tables directly in Action:

- `poles` stores physical frequencies unchanged.
- `tau_values` contains the combined scalar functions at the mesh nodes.
- `tau_values_reflect` is unsigned `C(beta-tau)`.
- `kernel` contains the existing TauKernelValues evaluated at `beta*poles`.

The symbolic adapter establishes statistics from the supported operator
expressions; the matrix adapter receives them. Preserve the current supported
bosonic dynamical-interaction model. Audit zero, odd and mixed-parity inputs
before applying the existing parity helper wholesale: it rejects zero/mixed
expressions. Do not hide a change of accepted operator classes in this refactor.

Delete ArcCatalog and its kind/family types. Move the needed Statistics enum to
a small diagram statistics header. `diagram::Arcs`, operator references and
barred-contraction descriptions remain structural inputs indexing Action data.
The first `n_hyb` flavours identify single-particle operators; arbitrary
interleaved family order is no longer a model-construction feature.

## One mesh, all DLR functionality

The local TRIQS/cppdlr source audit confirms that Action can retain only
`tau_mesh` as its DLR state. The mesh already owns/shares a full imtime_ops object:

```cpp
auto const &itops = action.tau_mesh.dlr_it();
auto nodes = itops.get_itnodes(); // const view, not an owning node array
auto beta = action.tau_mesh.beta();
auto rank = action.tau_mesh.size();
```

`dlr_imtime::dlr_it()` returns a const reference. All operations used by the
current evaluators are callable through it: `vals2coefs`, `convolve` (including
TIME_ORDERED) and `reflect`. Kernel preparation also obtains nodes/rank through
this object. There is no missing functionality requiring a separately owned
`itops`, and no need to add forwarding convolution methods to Action.

The names currently obscure different roles: `tau_mesh.dlr_it()` returns the
operations object, whereas the evaluator member called `dlr_it` is a copied
node vector. In both evaluators that member is initialized but never read by the
implementation. Remove it. Replace owning `itops` members with local const
references and eliminate the extra evaluator mesh once Action owns it. Use
`action.tau_mesh` when constructing output Green's functions.

Mesh copies already share their internal DLR tables through a shared pointer;
the current evaluator mesh and ArcData mesh are not independent expensive DLR
reconstructions. The extra by-value imtime_ops is a separate owning object,
including its matrices/caches, and the node member is an unnecessary copy.
Prefer `auto const &itops`, not `auto itops`, when borrowing the mesh operations.
Do not add reference members merely to shorten access: local references avoid
copy/move lifetime complications.

Three details must be retained in implementation:

1. **Lazy caches.** cppdlr's const reflection/convolution methods initialize mutable
   cached matrices on first use. Mesh copies share those caches. This permits
   const use but does not establish thread-safe concurrent first initialization.
   If a shared mesh will be used by concurrent evaluators, initialize the needed
   caches serially before that use. The installed API exposes `reflect_init()`
   and `tconvolve_init()`; do not repeatedly reinitialize them in hot loops.
   The current mutable-workspace evaluators themselves are not concurrent-call
   objects. No additional DLR copy is needed for ordinary sequential use.
2. **The old dense low-level constructor.** It accepts an independent imtime_ops
   but reconstructs a mesh from lambda/epsilon. Those objects are not guaranteed
   to have identical nodes for custom inputs. Prefer a mesh-based direct
   constructor and migrate internal tests to supply their actual mesh. If the
   old C++-only overload remains, validate grid compatibility rather than silently
   substitute a reconstructed grid. The installed mesh has no public constructor
   that imports an arbitrary imtime_ops; exact custom-grid import would be a
   separate TRIQS API issue, not a reason for duplicate state in normal use.
3. **Units and wrappers.** `get_itnodes()` returns cppdlr's relative-time nodes;
   mesh iteration returns physical times. Do not substitute one for the other
   in kernel formulas. Dense currently exposes tau_mesh/itops/dlr_it properties
   through generated bindings. Preserve read access via derived properties if
   needed, without duplicate member storage; check conversion/lifetime behavior.
   Independent mutation of these properties cannot remain a supported way to
   alter an otherwise immutable evaluator. Record that compatibility detail.

Keep the contiguous copy at block-sparse reflection of a workspace slice:
`itops.reflect(nda::make_regular(result))`. Removing DLR redundancy must not undo
this existing strided-input fix. Scope the cleanup to evaluator/Action state;
manual reference routines may continue accepting imtime_ops arguments.

## Boundaries and implementation sequence

Use ordinary overloaded setup helpers for dense and block-sparse storage. A
single template parameter does not require traits, policy hierarchies or runtime
backend machinery. Keep Action internal to the existing Python evaluator types.
Binding/sign helpers receive the fields they use, not the entire Action.

The representation-specific matrix setup still deserves a separate review:
current dense fundamental-operator construction supports multiple atomic
subspaces, unlike its dynamical-operator route. Fundamental block-sparse grouping
uses matching connection rows; dynamical grouping allows compatible missing
blocks. Preserve these capabilities and admissible coefficients during Action
adoption. Do not silently route all operators through a more restrictive path.

A reviewable sequence is:

1. **Introduce numerical Action and construction adapters.** Test scalar fields,
   operators, explicit metadata and bars from the same coefficients. Cover both
   atom_diag and numerical-only inputs, with no retained atom_diag or separate
   persistent operator list. Existing classes may remain temporarily as comparison
   references, but do not add duplicate model storage to an evaluator.
2. **Adopt Action in the dense evaluator.** Preserve Python constructors and direct
   C++ test construction; move FSet/scalar ownership into Action. Keep current
   basis ordering, contractions and numerical entry points.
3. **Adopt Action in the block-sparse evaluator.** Preserve custom groupings,
   support behavior and dimensions, including all-zero sectors. Move quartet
   ownership, update bind inputs, and retain matrix-only tests.
4. **Consolidate DLR state.** Route transformations, convolutions, reflection and
   output meshes through Action's mesh. Remove owning itops/node members and
   settle the old low-level grid adapter and derived Python properties. This is
   separate from changes to operator grouping or contraction arithmetic.
5. **Remove obsolete concepts and close validation.** Delete ArcData/ArcCatalog
   and repeated coefficient assembly; migrate their tests to Action and diagram
   sign tests. Check wrapper API, real/complex construction, direct numerical
   construction, copy/move behavior and the complete CPU/Python suite.

Only afterwards consolidate remaining operator-resolution/grouping algorithms
as an independently reviewed change. No GPU implementation or MPI partition API
is part of this plan. Future GPU execution can use Action's numerical arrays
without CPU enumeration of every backbone path; performance targets remain
high-end cards such as GH200, not the local RTX A4000.

Use focused code commits and separate `work_log_action.md` checkpoints, with
terse house-style messages. Each checkpoint explains the change, preserved
contracts and relevant validation. Reuse existing convention, mixed-interaction,
symmetry, numerical-zero, strided-input, dense/block-sparse agreement and Python
solver tests. Add tests only for the new construction/ownership/grid contracts.

## Sources reviewed

Repository: `c++/triqs_xca/{dense,block_sparse}/diagram_evaluator.{hpp,cpp}`,
`arc_data.{hpp,cpp}`, `hyb.cpp`, both operator/dynint setup implementations,
`dense/fset.hpp`, `block_sparse/block_op.hpp`, `atom_diag.hpp`, and generated
`python/triqs_xca/_triqs_xca.wrap.cxx` properties. Direct-construction coverage
includes sparsity_invariance, numerical_zero, sym_set_mask and evaluator_conventions.

Installed dependency sources checked on 2026-10-24:

- `/home/hugstr/apps/triqs/include/triqs/mesh/dlr_imtime.hpp`: operations accessor
  at line 274, physical-time conversion at 248–253, shared ownership at line 391.
- `/home/hugstr/apps/triqs/include/cppdlr/dlr_imtime.hpp`: const transforms at 103,
  reflection at 265, convolution at 311, node view at 721, and lazy cache storage
  near 932–943. Public cache initialization methods are at 833 and 901.

This revision was source-reviewed; no evaluator code, builds or benchmarks were
changed or run for the documentation update.

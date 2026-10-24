# Action refactoring plan

Revised and reviewed 2026-10-24 on `dev_backbone`. This replaces the initial
ownership sketch with the subsequent discussion. Planning only: implementation
has not been requested in this revision.

## Agreed structure

Introduce a flat `triqs_xca::Action` in `c++/triqs_xca/action.hpp`. Remove the
`ArcData` and `ArcCatalog` concepts rather than putting them inside Action.
Keep the existing numerical arrays and their conventions, with these names:

| Action member | Meaning and ownership |
| --- | --- |
| `tau_mesh` | Owned imaginary-time mesh |
| `kernel` | Owned `TauKernelValues`, computed from mesh and poles |
| `atom_diag` | Const reference to the caller's atomic realization; never copied |
| `ops` | Owned list of second-quantized operators, one per combined flavour |
| `poles` | Owned physical frequencies, not beta-scaled |
| `pole_coeffs` | Owned combined pole coefficient tensor |
| `tau_values` | Owned scalar functions on `tau_mesh` |
| `tau_values_reflect` | Owned unsigned time reflection `C(beta-tau)` |
| `statistics` | Per-operator fermion parity, established during construction |

These are logical member names; external access must not permit independent
mutation of arrays that would invalidate derived tables or cached F-bars.
Beta and flavour/pole counts are derived from these members, not independently
editable model properties. Mesh-dependent numerical workspaces remain evaluator
state.

The evaluator owns Action, F-bar storage and its workspaces. Action is constructed
inside the existing evaluator constructor. The propagator is a separate input.
Action owns no numerical operator matrices: dense `FSet` and block-sparse
`BlockOpSymQuartet` remain the representation-dependent operator/F-bar storage.
There is no additional persistent numerical model beside Action and F-bar.

The caller guarantees that atom_diag outlives the evaluator and remains unchanged
while its derived state is used. Do not copy atom_diag, move it out of the solver,
or introduce shared ownership. Action's lifetime is the evaluator's lifetime.
Replace the evaluator when fitted model data changes, as the solver already does.

## One operator list and one coefficient tensor

Keep **`dynint_ops`** as the existing constructor argument name. Construct:

```text
h = atom_diag.get_fops().size()
d = dynint_ops.size()
ops = [c_0, ..., c_(h-1), dynint_ops[0], ..., dynint_ops[d-1]]
```

Generate the annihilation-operator expressions from atom_diag in its fundamental
operator order. Store one expression per flavour, not separate entries for both
an operator and its adjoint. Representation construction derives the adjoints.
Do not retain another `dynint_ops` vector after assembling `ops`; its entries
are available as the tail of `ops`.

This is the indexing contract:

```text
ops[a]                   operator for flavour a
statistics[a]            its fermion parity
pole_coeffs[p, a, b]      coefficient of ops[a]^dag ... ops[b] in a forward arc
```

Backward arcs still use `diagram::ArcOrientation` and
`BarredOperatorContraction`, including their index reversal and reflection sign.
No new sign convention or factor of one half is introduced.

Continue the existing `hyb::get_extended_coefficients` construction, performed
once per Action:

```text
pole_coeffs[p] = [ hyb_coeffs[p]       0          ]
                [      0       dynint_coeffs[p] ]
```

Both inputs share the same physical pole list. Off-diagonal entries within the
retarded block remain supported subject to the selected representation's
restrictions. For example, with `ops = [c_up, c_down, n_up, n_down]`, the forward
coupling of `n_up^dag` and `n_down` occupies `pole_coeffs[p,2,3]`.

Inputs remain already fitted poles and coefficients. The existing Python joint
fit and subsequent coefficient split need not change. No refitting, pole merging
or separate per-operator pole sets are introduced. The original input coefficient
blocks are not retained alongside `pole_coeffs` inside the evaluator. Const
slices suffice when setup needs one block.

Move ArcData's scalar precomputations directly into Action: construct
`tau_values`, its unsigned reflection, and `kernel`. Apply beta scaling only at
the dimensionless cppdlr kernel calls. Bosonic operator statistics do not change
the fitted scalar-kernel representation or normalization.

## Remove kind and families

There is no `ArcKind`, `ArcFamily`, catalogue, or stored family list in the final
model. Current numerical consumers need statistics and the external
single-particle flavours, not family objects.

The first `h` entries identify the fundamental annihilation operators. Derive
that range from atom_diag; never select fundamental operators merely by scanning
for fermionic statistics. A general odd-parity operator is not necessarily a
single-particle field. This ordering convention deliberately replaces the
catalogue's ability to describe arbitrarily interleaved families.

`diagram::Arcs` remains: it records the chosen directions, pole indices and
retained flavours of a particular diagram. These indices refer to Action's
arrays. `diagram::OperatorReference` and `BarredOperatorContraction` also remain.
Move the still-needed `Statistics` enum out of the deleted catalogue header into
a small diagram statistics header, updating includes without changing its meaning.

Block-sparse operator symmetry sets remain in F-bar construction. They describe
matrix connectivity, not physical interaction families, and need not follow
flavour order. Preserve their existing global-flavour mappings.

## Preserve the Python API and make ownership concrete

Existing Python-facing constructors remain unchanged, including real/complex
atom_diag overloads, argument names and optional dynamical-interaction arguments.
For example, existing calls continue to look like:

```python
evaluator = DenseDiagramEvaluator(
    poles, hyb_coeffs, mesh, ad, dynint_ops, dynint_coeffs)
# BlockSparseDiagramEvaluator accepts the corresponding existing arguments.
```

Internally the sequence is:

```text
borrow atom_diag; construct owned Action
    -> construct operator matrices and F-bars from Action
    -> initialize support information and numerical workspaces
```

Action is an internal member (excluded from automatic Python wrapping), not a
temporary whose arrays are borrowed after construction. Declare it before state
that depends on it. Do not cache pointers into its members
unless evaluator copy/move behavior explicitly maintains those pointers. Prefer
access through the owning evaluator and call-local bound views. Preserve existing
Python-visible properties, such as beta/counts, through compatible access where
needed; unchanged constructors alone are not the entire API compatibility check.

The existing evaluator classes are non-templated but accept both
`atom_diag<false>` and `atom_diag<true>`. A concrete implementation recommendation
is a non-templated Action with a closed two-alternative reference holder:

```cpp
using AtomDiagRef = std::variant<
    std::reference_wrapper<triqs::atom_diag::atom_diag<false> const>,
    std::reference_wrapper<triqs::atom_diag::atom_diag<true> const>>;
```

This represents a const reference to either existing atom_diag type, never an
owned atom_diag or nullable model. Visit it in setup operations that need the
atomic realization. Numerical loops read Action's non-templated fields directly;
there is no variant dispatch per backbone contribution. No abstract base class,
virtual model interface or new evaluator template hierarchy is needed.

The lifetime guarantee is supplied by the caller, including Python callers.
The solver already retains its atomic realization; document and check its
construction/replacement order. Binding regeneration must not introduce a
conversion temporary or copy in place of the referenced object. Do not add a
new ownership mechanism to compensate for lifetimes the user has guaranteed.

Existing correlator methods separately accept atom_diag. Preserve those
signatures and their current use of the supplied argument and compatibility
checks in this refactoring. Do not silently ignore that argument or newly require
pointer identity. Any later simplification to use only Action's reference is a
separate API decision. Correlator operator matrices remain separate from `ops`.

## Operator construction and binding boundaries

The common destination is operator construction from `ops`, atom_diag and
`pole_coeffs`, without the F-bar contraction needing to know which entries came
from hybridization. FSet and BlockOpSymQuartet already contract supplied matrices
and coefficients. Most special handling lives in constructing those matrices
and their symmetry groups.

Separate adopting the common inputs from unifying every implementation detail:

- Dense fundamental-operator construction currently handles multiple atom_diag
  subspaces. The dynamical-operator path currently requires one subspace. Routing
  every operator through the latter would be a regression; retain the existing
  capability while a general matrix-resolution path is developed.
- Fundamental block-sparse grouping matches connection rows. Dynamical grouping
  accepts compatible rows with missing blocks. Preserve the existing partitions
  and admissible coefficients initially; replacing both grouping algorithms is
  a separate, independently validated step, not a consequence of moving data.
- Block-sparse single-target/injective connection requirements and restrictions
  on cross-symmetry-set coefficients remain representation-specific. Action
  construction must not impose them on models meant for dense evaluation.

Helpers continue to receive the narrow inputs they use: sign preparation takes
statistics; tau-kernel binding takes `kernel`; endpoint binding takes the scalar
tables and operator data; backbone binding takes propagator and F-bar views.
Do not pass all of Action through shared diagram code merely because it is now
available. Update bind functions that explicitly mention ArcData to take the
required fields, rather than introducing a replacement persistent view container.

Bound CPU operands remain temporary. Action does not prescribe enumerating or
transferring every backbone path. GPU storage and execution strategies remain
future work, targeting high-end cards such as GH200 rather than the local RTX
A4000. No MPI work-partition API or GPU benchmark work belongs to this refactor.

## Review findings and migration checks

The review identified these concrete traps and incorporated them into the plan:

1. **A templated Action cannot be a single direct member of today's non-templated
   evaluator.** Use the small real/complex reference holder above, without copying
   atom_diag or duplicating the numerical fields across variant alternatives.
2. **Some C++ tests construct evaluators without atom_diag.** In particular,
   sparsity-invariance, numerical-zero, symmetry-mask and convention tests supply
   prebuilt operators. Audit these before switching constructors. Use real
   atom_diag-backed fixtures where they describe physical models, keeping custom
   propagators and operator-grouping coverage. Keep genuinely synthetic checks
   at the operator/numerical boundary when they cannot describe an Action.
   Do not fabricate atom_diag merely to fill a field, introduce a half-valid
   Action, or retain ArcData/ArcCatalog as a permanent test escape route. The
   exact test adaptation is an implementation checkpoint before deleting the
   old low-level constructors; preserving coverage is required.
3. **Deriving parity has edge cases.** The current helper rejects zero and
   mixed-parity expressions, while current dynamical setup assumes bosonic
   operators. For nonzero supported even operators, derived statistics agree.
   Explicitly audit zero/odd/mixed inputs before applying the helper wholesale:
   do not accidentally reject previously supported zero operators or claim
   support for odd dynamical interactions merely by changing their sign flag.
   Any change to accepted operator classes requires its own documented decision
   and regression, rather than being hidden in the ownership change.
4. **Generic resolution can change basis ordering, support or summation order.**
   Compare against existing Fock-basis matrices and F-bars, including adjoints,
   non-Hermitian operators, multiple subspaces and missing blocks. Test numerical
   agreement at established tolerances; do not weaken tolerances to mask errors.
5. **Immutability includes the borrowed atom_diag.** A live reference is not
   sufficient if the caller changes its basis or partition after F-bar setup.
   State this invariant alongside the lifetime guarantee.

## Reviewable implementation sequence

1. **Flat Action and construction tests.** Implement the members, borrowed
   real/complex atom_diag, unified `ops`, combined coefficients and scalar tables.
   Compare with existing construction, including beta != 1 and empty dynint input.
   Old classes can remain temporarily for comparison during migration, but no
   evaluator should own duplicate old/new scalar model storage.
2. **Dense evaluator adoption.** Construct Action inside the existing public
   constructors; source scalar data/statistics from it and feed its common inputs
   to F-bar setup. Migrate affected low-level tests. Preserve existing dense
   matrix-resolution capabilities and Python-facing behavior.
3. **Block-sparse evaluator adoption.** Do the corresponding ownership change,
   bind-signature updates and test migration. Preserve symmetry grouping,
   support construction and coefficient admissibility while changing ownership.
4. **Remove the obsolete concepts and close compatibility checks.** Delete
   ArcData/ArcCatalog/family types and obsolete constructors once callers migrate.
   Replace catalogue-specific tests with Action ordering/statistics and diagram
   sign tests. Remove repeated coefficient assembly from setup, update wrappers
   as needed, and run the complete CPU/Python suite.
5. **Unify remaining operator-resolution machinery separately.** Review the
   actual duplication left by stages 2–3, then consolidate matrix construction
   behind the common operator inputs. Keep grouping-policy changes separate if
   they affect admissibility or work ordering. This stage must not be folded into
   the ownership commits merely to make the implementation appear uniform.

Each stage should be understandable from its code commit and a separate
`work_log_action.md` checkpoint: what changed, why, preserved contracts and
validation. Use the established terse commit style. Split a stage only where its
size prevents useful review; do not combine ownership, grouping changes and
unrelated naming cleanup in one commit.

Use the existing convention, interaction-contraction, mixed-interaction,
dense/block-sparse comparison, symmetry, correlator and Python solver tests.
Add focused ownership/indexing/matrix-equivalence tests where those contracts
are new. Validate real and complex atom_diag routes and wrapper signatures.
Documentation revision itself requires no build or numerical benchmark.

## Sources checked for this revision

- `arc_data.{hpp,cpp}` and `diagram/arc_catalog.{hpp,cpp}` under
  `c++/triqs_xca/`: scalar precomputation, metadata and removal targets.
- `dense/atom_diag.cpp`, `block_sparse/atom_diag.cpp` and both `dynint.cpp`
  implementations: matrix construction and different grouping policies.
- Both `diagram_evaluator.{hpp,cpp}` implementations: ownership, real/complex
  constructors, external-flavour selection and explicit correlator atom_diag.
- `c++/triqs_xca/operator_statistics.hpp`: zero/mixed-parity behavior.
- `python/triqs_xca/_triqs_xca.cpp` and `solver.py`: existing exposed signatures
  and solver construction pattern.
- `test/c++/block_sparse/{sparsity_invariance,numerical_zero,sym_set_mask}.cpp`
  and `test/c++/evaluator_conventions.cpp`: low-level constructor dependencies.

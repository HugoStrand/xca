# Action implementation work log

Persistent goal started 2026-10-24 on `dev_backbone`, following
[plan_action.md](plan_action.md), revision `6807f9a`.

The evaluator will own Action templated on its existing F-bar storage. atom_diag
is used only during construction; direct numerical construction remains supported.
ArcData/ArcCatalog are removed, and Action's mesh becomes the sole DLR state.
Python-facing constructor signatures and existing numerical conventions remain.

## Checkpoint 0: scope and starting state

Implementation is authorized. Planned checkpoints are Action construction,
dense adoption, block-sparse adoption, DLR consolidation, and removal/review.
Code commits and documentation checkpoints will remain separate.

Starting tree contains unrelated changes to `.gitignore` and untracked AGENTS
files, CUDA smoke test, Python cache/exp_order_conv and tools. Preserve them.
Previous full CPU verification passed all 75 CTest entries. No CUDA or benchmark
work is required for this refactor.

The first construction tests will cover physical poles, table equivalence,
operator metadata and rebuilding stale F-bars from the Action coefficients.

## Checkpoint 1: numerical Action construction

Code: `57177a6` — `[action] add numerical model storage`.

Added Action with flat scalar fields and representation-specific operator storage.
Dense and block-sparse factories support both atom_diag setup and numerical
operator inputs. atom_diag is never retained. A construction-only callback builds
storage from the Action's owned coefficients; it is invoked once, not stored or
used during evaluation. Numerical adapters rebuild bars instead of trusting
potentially stale input contractions. Explicit sparse dimensions are validated;
legacy dimension inference retains -1 for unseen sectors.

Added combined-coefficient overloads to the existing dynamical setup routines,
so Action can assemble coefficients once. Matrix construction and grouping are
unchanged. Invalid coefficient block shapes now throw rather than depend on
assertions. Statistics preserve the current fundamental-fermion/bosonic-dynint
input convention; no new operator class is silently admitted or rejected.

Action tests compare scalar data with the old implementation, check physical
poles and mixed/empty dynamical ordering, and deliberately reuse stale bars to
verify rebuilding. Full CPU build passed; all 76 CTest entries passed in 56.81 s
with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, parallelism 2. Python wrappers
unchanged. Logs: `/tmp/xca-action-stage1-{build,test}.log`.

ArcData/ArcCatalog and both evaluators remain at this checkpoint as comparison
references. Next: dense ownership migration, then block-sparse migration.

## Checkpoint 2: dense evaluator ownership

The dense evaluator now owns a const Action instead of separate FSet, ArcData
and ArcCatalog objects. Matrix contractions, scalar bindings and diagram signs
read that owner. Python constructors and generated wrappers are unchanged;
C++ tests that inspected Fset now inspect action.operators. DLR consolidation is
intentionally deferred to its own checkpoint.

Full CPU build passed; all 76 CTest entries passed in 56.85 s with the standard
single-threaded numerical libraries. Logs: `/tmp/xca-action-dense-{build,test}.log`.

## Checkpoint 3: block-sparse evaluator ownership

Block-sparse construction now delegates to one Action constructor. Operators,
scalar tables, statistics and sector dimensions live in that owner; support
metadata and mutable workspaces remain in the evaluator. Bind helpers receive
scalar views rather than Action or ArcData. Direct Action input permits explicit
sector dimensions; the new test retains a dimension-three sector on which all
operators vanish, and rejects a propagator of the wrong size there. Legacy
quartet adapters still infer unknown dimensions as -1.

Full CPU build passed. The full suite passed 75/76 entries; the sole failure
was a test matching the old constructor's diagnostic wording. Action diagnostics
now report expected and actual dimensions; the adjusted mask test and Action
tests both pass (0.35 s). Numerical results required no changes or tolerance
adjustments. Logs: `/tmp/xca-action-sparse-{build,test,focused}.log`.

## Checkpoint 4: one DLR owner

Both evaluators now use action.tau_mesh for transforms, convolutions, reflection
and output meshes. Removed their independent imtime_ops, node arrays and mesh
members. Dense construction also delegates to one numerical Action constructor.
The old C++ dense adapter validates both real-frequency and imaginary-time nodes
before accepting its reconstructed mesh. Direct Action construction accepts an
explicit mesh, including symmetric grids. Block-sparse strided reflection still
makes a contiguous temporary before calling cppdlr.

Python constructor signatures remain unchanged. Dense tau_mesh, itops and dlr_it
reads remain available as derived, read-only properties; operation/node reads
return copies on demand. Independent setters are intentionally removed. Numerical
methods borrow the mesh operations directly and never call these copying getters.
Shared mesh lazy caches retain the sequential-use constraint documented in the
plan; no concurrent-first-use guarantee has been added.

New tests cover grid rejection, explicit mesh adoption, evaluator copies surviving
source destruction, and Python property reads/immutability. Full CPU build and
all 76 CTest entries passed (56.87 s). Logs: `/tmp/xca-action-dlr-{build,test}.log`.


## Checkpoint 5: removal and final review

Code: `85bcee0` — `[action] remove arc data and catalogue`.

Removed ArcData, ArcCatalog and their kind/family metadata. Structural diagram
headers include only Statistics. Scalar and sign tests now use Action or the
specific numerical tables/statistics they exercise; manual references reuse the
existing evaluator Action. The backbone benchmark fixture was adapted and
compiled, without running benchmarks or changing results.

Final review checked ownership, lifetime, grid compatibility, scalar conventions,
coefficient assembly and wrapper changes. There are no retained atom_diag or
symbolic-operator references, no second persistent ordinary-operator list, and no
views from operator storage into its enclosing Action. Numerical adapters rebuild
bars from the owned coefficients. Support tables own their metadata. Explicit
sector dimensions include inactive sectors; legacy unknown dimensions remain -1.
Physical poles, unsigned reflected scalar values, explicit reflection signs and
existing operator-grouping restrictions are unchanged. No numerical tolerance was
loosened. No obsolete model-type references remain in source, tests or benchmarks.

Final CPU configure/build passed. All 76 CTest entries passed in 57.02 s, with
OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and test parallelism 2. The optional
backbone_cpu target also builds. Logs: `/tmp/xca-action-final-configure.log`,
`/tmp/xca-action-final-build.log`, `/tmp/xca-action-final-test.log` and
`/tmp/xca-action-benchmark-build.log`. No CUDA validation was part of this scope.

The goal is complete. Remaining design work is deliberately separate: unifying
operator-resolution/grouping algorithms and designing GPU data layouts. Shared
DLR caches require serial first initialization before concurrent sharing, as
noted at checkpoint 4. Python constructors are unchanged; the three dense DLR
properties are read-only compatibility views (copies on operation/node access).

## Review order

Each code commit is followed by a separate documentation checkpoint.

| Code commit | Change |
|---|---|
| `57177a6` | Numerical Action and construction adapters |
| `37f0b1f` | Dense ownership migration |
| `4016f3b` | Block-sparse ownership and explicit sector dimensions |
| `6d00e39` | Single DLR owner and compatibility properties |
| `85bcee0` | Remove obsolete model types and migrate remaining consumers |

Unrelated initial worktree changes and untracked files were preserved.


## Follow-up: direct kernel access (2026-10-24)

Code: `b1b8c9a` — `[diagram] simplify kernel access`.

Removed VertexKernelView, KernelView and TauKernelViews, together with the root
bind/tau_kernel files. Both evaluators now apply the existing Diagram recipes
directly against action.kernel. Numerical helpers accept the recipe, pole indices
and kernel tables; they do not depend on the Action template. Table selection
happens outside the time/matrix loop. The scalar prefactor is still calculated
once per diagram and reused across block-sparse paths. Backbone, endpoint and
correlator operand binding are unchanged.

Renamed TauKernelValues to TauKernel, its header to tau_kernel.hpp, and its
factory to make_tau_kernel. Numerical parameters and the factory result are
named kernel. The class comment identifies the contents as precomputed tables;
logical placement and prefactor types retain their diagram namespace and names.
The Action plan has been updated to reflect this naming.

Review confirmed that reflection reverses only vertex argument signs, zero
recipes remain identities (including order-one diagrams with no internal poles),
and edge factors are applied in the original pole-slot order. Existing kernel
tests now check numerical application rather than retained pointers. Additional
coverage detects factor reordering and premature product formation through
avoidable overflow/underflow. Changed tables are read directly, without rebinding.

The initial refactor passed all 76 CTest entries (57.10 s). After the naming
refinement, configure/build and all 76 CPU/Python entries passed again (57.16 s),
with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and test parallelism 2. Generated
Python bindings are unchanged. The backbone_cpu benchmark target compiles; its
preparation row is now sign_and_kernel_prefactor, documented separately from the
older binding measurement. No benchmarks were run and no speedup is claimed:
removing bound containers trades their allocations for repeated direct lookups
and scans of an edge's placement row.

Final logs: `/tmp/xca-kernel-renamed-configure.log`,
`/tmp/xca-kernel-renamed-build.log`, `/tmp/xca-kernel-renamed-test.log`, and
`/tmp/xca-kernel-renamed-benchmark-build.log`. Unrelated worktree changes remain
untouched.

## Follow-up: consolidate numerical helpers (2026-10-25)

Code: `db4c851` — `[action] consolidate numerical helpers`.

Moved Statistics to action.hpp in the triqs_xca namespace: parity describes the
model's operators and external observables, rather than diagram geometry. Moved
TauKernel, its numerical application routines and get_extended_coefficients
into action.hpp/action.cpp. Updated both evaluators, construction adapters,
binding helpers, tests and the backbone_cpu benchmark to use these declarations.
Removed the root tau_kernel.hpp, tau_kernel_ops.hpp/.cpp, hyb.hpp/.cpp and
diagram/statistics.hpp headers/sources.

Logical kernel placement and prefactor recipes remain in diagram/tau_kernel.*.
The Action header forward-declares the two recipe types used by its numerical
helpers; their definitions are included only in action.cpp. This avoids a
circular include. A consequence of consolidation is that diagram headers using
Statistics now include the numerical Action header; they do not retain an Action
or acquire additional numerical state.

Retained the coefs2vals overload accepting existing imaginary-time operations,
now in triqs_xca. Action construction and manual/reference test setup use it to
evaluate coefficients at fitted physical poles, which can differ from the mesh's
native DLR poles. Therefore cppdlr::imtime_ops::coefs2vals is not a replacement.
Removed the unused beta/Lambda/eps overload that constructed a separate grid.
The retained helper's strided-input handling and all numerical algorithms are
unchanged. No compatibility aliases or forwarding headers were added; C++ callers
use the new header and namespace, while generated Python bindings are unchanged.

CPU reconfiguration and build passed. All 76 CTest entries passed in 57.17 s,
with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and test parallelism 2. The optional
backbone_cpu target also builds; no benchmark timings or CUDA validation were
part of this cleanup. Logs: `/tmp/xca-action-files-configure.log`,
`/tmp/xca-action-files-build-final.log`, `/tmp/xca-action-files-benchmark-build.log`
and `/tmp/xca-action-files-test.log`. Unrelated worktree changes were preserved.

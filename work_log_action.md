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

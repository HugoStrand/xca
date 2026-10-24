# Evaluator convention cleanup

Date: 2026-10-24. Branch: dev_backbone.

The user authorized two consistency fixes before introducing Action: use the same
meaning for reflected coefficient values, and accept physical poles at every
evaluator constructor. This work deliberately leaves operator storage, F-bar
construction, diagram/bind organization and Action naming unchanged.

## Step 1: reflected coefficient values

Previously dense ArcData stored -C(beta-tau), while block_sparse stored
C(beta-tau) and applied the orientation sign during endpoint contraction.
These compensated conventions produced matching results but prevented the two
evaluators from sharing the same numerical data without special handling.

ArcData now always stores C(beta-tau), with no orientation sign, transpose or
conjugation. Its configurable reflection_sign constructor argument is removed.
The dense self-energy endpoint contraction explicitly multiplies by
ArcOrientation::reflection_sign(), matching block_sparse. The existing coefficient
row/column exchange for backward orientation is retained.

F_bars_refl is unchanged in both implementations: it is a materialized backward
operator contraction and already includes its prescribed minus sign through the
shared BarredOperatorContraction. Diagram-order, direction and permutation signs
also retain their existing meanings and application points.

A new evaluator_conventions regression uses two fermionic operators, complex
nonsymmetric coefficient matrices, positive and negative poles, and beta=2.
It computes C(tau) and C(beta-tau) directly from exponentials, then constructs
first-order forward and backward results by explicit matrix multiplication.
Both evaluators must match each direction individually; the dense paired result
must match their sum. This catches missing/doubled signs, transposed indices and
unwanted conjugation without relying only on agreement between implementations.
Existing tests independently verify both F-bar contractions and mixed statistics.

Code commit: `554c419` — `[dense] align time reflection convention`.

Validation:

- CPU configuration and full build passed with binding regeneration enabled.
- The new focused regression passed, followed by all 75 CTest entries in 56.49 s
  (OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, CTest parallelism 2).
- The Python tree, including generated wrappers, is unchanged.
- git diff --check passed. No existing numerical tolerance was changed.

Logs: `/tmp/xca-conventions-reflection-{configure,build,test}.log`.
The new test temporarily passes beta-scaled poles to the legacy dense low-level
constructor; Step 2 will migrate that call together with the constructor contract.

## Step 2: physical poles at evaluator construction

The dense low-level constructor taking FSet previously accepted beta-scaled pole
arguments, unlike the atom_diag and block_sparse constructors. It now takes
physical_poles and passes them directly to ArcData. ArcData remains the single
conversion point: its stored poles are beta-scaled for kernel preparation. No
kernel formulas or coefficient normalizations change.

Five callers in block_sparse/sparsity_invariance.cpp passed the dimensionless
DLR nodes directly to dense construction; they now divide those nodes by beta,
just as the corresponding block_sparse calls already did. Two existing low-level
dense callers use zero poles and require no numerical migration. The Step 1
regression also now passes physical poles directly. Repository call sites were
audited; the Python-facing atom_diag constructors already used physical poles.

The common unit contract is documented in both evaluator headers and ArcData.
The low-level dense parameter is explicitly named physical_poles. This is a
semantic change to that C++ constructor: external users must divide any previously
beta-scaled inputs by beta. The constructor is not exposed to Python.

The new all_constructor_routes_accept_physical_poles regression checks beta=1.5
and beta=3 with positive and negative physical poles. It constructs dense and
block_sparse evaluators through both atom_diag and prebuilt-operator routes on
the same single-sector realization. It compares the stored poles, coefficients,
time functions, all four tau-kernel tables, and individual first/second-order
self-energy contributions covering both pole signs and all arc directions.

The test was verified to fail against the old dense constructor on pole, table
and result comparisons, then pass after the fix.

Code commit: `dba32ea` — `[dense] accept physical poles in all constructors`.

Validation:

- Full CPU build passed with binding regeneration enabled.
- The new constructor regression failed on the previous implementation and
  passed after the fix. The independent reflection/direction regression also passed.
- All 75 CTest entries passed in 56.75 s with the same thread settings as Step 1.
- The Python tree, including generated wrappers, is unchanged from `ee8b15d`.
- git diff --check passed. No existing numerical tolerance was changed.
- Unrelated initial tracked and untracked work was preserved.

Logs: `/tmp/xca-conventions-poles-{red-build,red-test,build,focused-test,test}.log`.
The red-test log intentionally records the failing baseline. Temporary logs may
be removed independently of this repository.

Reproduce after `source ~/env/xca-cpu.sh`:

```bash
(cd cbuild && bash ../cmake.sh)
cmake --build cbuild --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild --parallel 2 --output-on-failure
```

Both requested consistency fixes are complete. No Action, new data container,
F-bar refactor, GPU implementation or benchmark experiment was introduced.

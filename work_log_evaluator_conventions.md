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

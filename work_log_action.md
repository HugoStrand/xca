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

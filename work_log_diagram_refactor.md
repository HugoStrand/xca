# Diagram refactoring work log

## Starting checkpoint — 2026-10-25

Branch: `dev_backbone`, code baseline `ad9e470`.
Plan: [plan_diagram_refactor.md](plan_diagram_refactor.md).

The user approved arc objects, integrated backbone kernel recipes, a single arc
association per vertex, explicit endpoint operations and the recommendations
recorded in the plan. Implement these as separate reviewable code checkpoints,
with documentation commits following each checkpoint.

Baseline from the preceding change: all 76 CPU/Python CTest entries passed in
56.94 s; backbone_cpu compiles; generated Python bindings were unchanged.

Unrelated initial state to preserve: modified .gitignore; untracked AGENTS.md,
AGENTS.md.bak-20261009, test/c++/test_nda_cuda_smoke.cpp,
test/python/__pycache__/, test/python/exp_order_conv/ and tools/.

Status: plan recorded; implementation pending.

# Backbone reversal work log

2026-10-30. Branch dev_backbone; source baseline 522ad52.
Plan: plan_reverse_from_diagram_to_backbone.md.

The requested design makes enriched Arcs independent preparation inputs,
Backbone the complete dense instruction description, and one owning BackbonePath
the complete sparse description for self-energy and correlator work. Numerical
Action ownership and the functioning Python APIs remain unchanged.

Initial unrelated state: modified .gitignore; untracked AGENTS.md,
AGENTS.md.bak-20261009, test/c++/test_nda_cuda_smoke.cpp,
test/python/__pycache__/, test/python/exp_order_conv/ and tools/. Preserve all.

Plan reviewed independently by three agents (shared diagram preparation, sparse
execution, Action/F-bar boundary) and revised. The main clarifications concern
metadata validity, numerical-owner compatibility, propagator-mask rediscovery,
and exactly-once correlator normalization with cached segments. Interface-changing
checkpoints must also keep the benchmark consumer compiling.

Status: reviewed plan ready. Source implementation starts with numerical operator
conventions and enriched arc metadata; no source changes yet.

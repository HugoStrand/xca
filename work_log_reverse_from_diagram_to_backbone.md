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

## Checkpoint 1 — operator conventions and enriched arcs

2026-10-31. Source commit: [action] separate operator conventions.

The F-bar partial-contraction and operator-pair orientation descriptions now live
in action.hpp, beside the numerical data they describe. Diagram vertex preparation
uses direction flags directly. The old endpoint-sum descriptor is a temporary
bridge and will disappear when endpoints move into Backbone.

Arc preparation now resolves statistics and three-way pole signs from explicit
metadata or Action, validates selected indices, and replaces stale derived fields.
The existing flat digit order and numerical contraction formulas are unchanged.

Validation: CPU reconfigure and focused build passed; diagram/arcs (six cases),
interaction_contractions, backbone_indexing and backbone_parity all passed.
No Python binding changes; source diff --check clean. Logs:
/tmp/xca-backbone-reverse-1-{configure,build}.log.

Final plan-review clarification: full-path execution must reject selected ranges;
empty endpoint lists never imply identity and observable cardinality is checked.

Next: resolve Backbone instructions and migrate dense/sparse consumers, preserving
the specialized sparse completion wrappers only through the transition.

## Checkpoint 2 — self-contained Backbone

2026-10-31. Source commit: [diag] evaluate prepared backbones.

The runtime Diagram wrapper and diagram/arc.hpp are gone. Preparation now follows
Action metadata -> enriched Arcs, then Topology + Arcs -> Backbone. Vertices and
kernel factors carry resolved flavour/pole indices. Backbone owns endpoint
geometry, both permutation-sign cases, internal direction sign and ordered scalar
normalization terms. No evaluation lookup needs Topology or Arcs.

Dense numerical entry points consume Backbone; ordinary/barred FSet access is
numerical and no longer depends on diagram headers. Native dense correlators
borrow observable arrays. Sparse discovery and its temporary binding layer also
consume Backbone, preparing the next path-ownership change.

This checkpoint changes both evaluators together because removing Diagram and
arc-relative kernel instructions changes their shared interfaces. The sparse
completion records and numerical-view binding remain deliberately unchanged in
structure; their removal is a separate checkpoint.

Tests retain literal indexing/parity fixtures and now prove prepared dense
self-energy/pair/correlator evaluation after construction inputs are destroyed.
Repeated-pole normalization is checked against a finite arithmetic example that
would overflow if factors were regrouped. Independent cross-reviews of dense and
sparse numerical migration found no actionable regression.

Validation: full CPU build and backbone_cpu compilation passed; all 77 CPU/Python
CTest entries passed in 57.51 s. Generated Python binding changes are docstrings
only. No benchmark measurements were run. Source diff --check clean. Logs:
/tmp/xca-backbone-reverse-2-{configure,build,test,benchmark-build}.log.

The next checkpoint uses one generic owning BackbonePath for segments and complete
contributions. Discovery will own one streamed workspace per Backbone call rather
than expose a caller workspace that initially has no recipe. This simplifies the
API, at the cost of no longer amortizing its small array allocations across
different assignments; no performance claim is made for that tradeoff.

# Diagram index API cleanup work log

## Plan checkpoint — 2026-11-02

Starting source: `c4589ac` on `dev_backbone`.
Plan: `plan_diagram_index_api_cleanup.md`.

The agreed boundary is one topology per Python evaluator call, returning the
current rank's contribution, and one prepared Backbone per local C++ numerical
call. Python reduces completed sums. Single-index, vector and interval selections
will no longer be evaluator API; fixture indexing remains test-only.

Two independent read-only reviews checked MPI/paired scheduling and numerical
test migration. The revised plan explicitly preserves production symbolic-input
validation and complete-result public Solver helpers, checks MPI initialization,
and gives dense paired evaluation its own compact enumeration. No production
source changed during planning. No new benchmarks are requested.

Keep source and work-log commits separate, use terse house-style subjects,
preserve unrelated user changes, and do not push.

## Compact pair enumeration — 2026-11-02

Source checkpoint: `[diag] enumerate self-energy pairs` (immediately preceding
this documentation commit).

Added a compact pair count and decoder alongside the existing arc enumeration.
The decoder fixes the special arc to backward and decodes only internal arc
choices. It never forms an ordinary self-energy index or the twice-as-large
ordinary count. Existing ordinary enumeration order is unchanged.

Six new tests check independent digit expectations, explicit Cartesian coverage,
opposite-endpoint coverage, first-order/empty cases, invalid metadata and sampled
large counts. The large-count case decodes 2^62 pairs' sampled indices even though
the corresponding ordinary count overflows a signed 64-bit long.

Validation: CPU reconfiguration and focused build passed. diagram/arcs,
diagram/backbone and diagram/pairs passed (3/3 CTest entries, 0.05 s).
Logs: /tmp/xca-index-pairs-{configure,build,test}.log. Diff --check passed.

Further review sharpened the MPI checks: require a real three-rank CTest with a
timeout/resource declaration, assert local ownership and count Python reductions.
Fixture migration and topology orchestration are in progress. The Python edits
are not yet committed; source was frozen during the focused build.

## Topology API and local execution — 2026-11-02

Source: `d6ea952` `[diag] distribute topology work`.

Both evaluators now expose only whole-topology computation to Python. These calls
validate inputs, query the initialized world communicator and stream this rank's
work through the existing local numerical methods. They perform no collectives.
Paired dense evaluation uses the compact decoder directly, so every assigned
ordinal evaluates one pair. No ordinary-direction ownership filter remains.

Removed evaluator scalar/vector/interval/count/print adapters, native
topology-taking conveniences and FlatIndexSelection. reset() remains C++ only.
Backbone/BackbonePath execution and contraction arithmetic are unchanged. Ordinary
counting now recognizes empty internal choice spaces before irrelevant powers
can overflow, matching the new pair counter.

Python accumulates all requested topologies/orders and reduces at the public
complete-result boundary. Public Solver single-topology helpers still return a
complete sum; internal loops call the rank-local evaluator directly. Updated the
dimer benchmark caller accordingly. Regenerated bindings remove the old overloads.

Historical numeric fixture indices live in test_utils/prepared_evaluators.hpp,
which decodes them to Backbones and performs local execution. Invalid symbolic
observable/layout tests still exercise production APIs. Logical-Arcs lifetime
tests and independent analytic references were retained; paired tests now compare
explicit opposite endpoint Backbones, including both starting orientations.
The fixture migration and API removal were combined in one building checkpoint
because production validation tests must switch to the new topology-only
correlator signature at the same time. The diff mostly removes forwarding code;
there is no new numerical hierarchy or scheduler framework.

backbone_cpu now measures bounded prepared calls, with distinct new labels and
an updated generic README. Existing measurement files are untouched. It was
compiled, not benchmarked.

Validation: full CPU/Python build and binding regeneration passed. The first
build found obsolete selected-pair fixture helper calls in flat_index_range;
these were replaced with direct prepared pair execution before the successful
build. The integration workspace passed 82/82 CTest entries in 59.33 s, including
the dedicated C++ MPI test committed next and the serial/three-rank Python tests.
Python checks public results, first-order idle ranks and exactly one reduction
phase per complete result, independent of topology/order count.

Logs: /tmp/xca-index-api-configure.log, /tmp/xca-index-api-build-2.log,
/tmp/xca-index-api-mpi-benchmark-build.log, /tmp/xca-index-api-test.log.
Source edits were frozen during builds. Diff --check and a tracked Python/notebook
audit for removed count/split APIs passed. Unrelated user work remains untouched.

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

## MPI validation and final review — 2026-11-02

Source: `71501fb` `[test] check distributed diagram sums`.

Added an explicitly MPI-initialized C++ test executable, registered both serially
and with three ranks. It compares rank-local self-energy, SPGF and bosonic
correlator results to prepared numerical references, and compares compact pairs
to explicit opposite endpoint Arcs. It accumulates several topologies before
reducing and checks first-order empty ranks. Rank-zero-only evaluator calls
followed by a barrier verify that topology evaluation performs no collective.
Malformed topology, propagator, observable parity and atom_diag layout errors
are rejected even by idle ranks; subsequent valid calls recover correctly.

Independent adversarial review found no blocking numerical/ownership/reduction
issues. It requested explicit assertions that MPI runs really have three ranks
and observable-layout checks on idle ranks. Both were added before final
validation. MPI CTests declare PROCESSORS 3 and a 180-second timeout. Prepared
tests remain independent of MPI work assignment, and pair oracles use ordinary
Backbone evaluation rather than the paired kernel.

Final acceptance: 82/82 CTest entries passed (59.33 s), including mpi_evaluator,
mpi_evaluator_3ranks, Py_mpi_evaluation and Py_mpi_evaluation_3. Full CPU/Python
build and backbone_cpu compilation passed. No numerical tolerances were loosened.
No benchmark timings, GPU runs, pushes, or history rewrites were performed.

The implementation intentionally uses the existing world communicator and equal
counts as its initial partition. Topology generation stays in Python. Future
multi-topology scheduling/custom communicators can change orchestration without
restoring public linear-index selection. Dense scratch/reset behavior remains
unchanged at the local prepared interface. The persistent goal is complete.

## Dense properties and scratch storage — 2026-11-02

Source: `6eaf4d7` `[dense] hide evaluator internals`.

Removed the unused dense evaluator compatibility getters for tau_mesh, itops and
dlr_it. Numerical code already uses action.tau_mesh; the constructor consistency
test now reads that mesh directly, and the Python compatibility assertions were
removed. No production Solver or benchmark callers required changes.

Made T, U, GKt, Tkaps and Tmu private. Sigma remains public for the C++ prepared
self-energy interface, with C2PY_IGNORE to exclude it from Python. Regenerated
bindings remove all dense properties. The existing invalid-operation test now
checks that each rejected call preserves Sigma and that a subsequent valid call
accumulates correctly, without inspecting temporary arrays. Numerical kernels,
reset behavior and the block-sparse API are unchanged.

An independent read-only usage audit and final diff review found no remaining
consumers or blockers. Full CPU/Python build and binding regeneration passed;
all 82 CTest entries passed in 58.23 s, including serial and three-rank MPI tests.
Diff --check passed. Logs: /tmp/xca-dense-api-encapsulation-{build,test}.log.
Unrelated user changes were preserved; source and this log remain separate commits.

## Dense type aliases — 2026-11-03

Source: `5f1ac9b` `[dense] use evaluator type aliases`.

Adopted the user's added dlr_imtime, atom_diag, opt_t, Backbone and BackboneRange
names throughout DiagramEvaluator declarations, definitions and explicit
instantiations. Changed namespace using-declarations to valid class type aliases,
with a bool alias template for atom_diag. The existing gf_t/gf_vt aliases now use
dlr_imtime, and self-energy return types use gf_t. The free compatible_mesh helper
keeps namespace-qualified types. Evaluation and storage are unchanged.

Full CPU/Python build and binding regeneration passed. The generated wrappers
retain all four real/complex atom_diag constructor variants and both correlator
overloads. All seven focused CTest entries passed in 3.92 s: action, the three
dense tests, Py_api, Py_one_time_correlator and Py_dense_manual_NCA_OCA. Independent
read-only review and diff --check passed. Logs:
/tmp/xca-dense-aliases-{build,test}.log. No tests or numerical code were changed.

## Dense array, vector and complex aliases — 2026-11-03

Source: `08d6cf3` `[dense] use array aliases`.

Applied the user's additional arr_t, arr_vt, arr_cvt, vector and dcplx names to
member storage, signatures, definitions and explicit instantiations. Array aliases
take the element type and rank and retain NDA's default layouts and ownership;
vector aliases std::vector with its default allocator. Trailing return types in
out-of-class definitions keep the class aliases in scope. nda::vector_const_view
remains explicit because its vector algebra differs from a rank-one array view.

The previous source commit picked up four additional namespace-style using
declarations that arrived after its build. That intermediate header was not the
validated version. This follow-up replaces them with valid template aliases and
validates the complete result. Source hashes were checked after the build and
before staging to avoid repeating that mismatch.

Full CPU/Python build and binding regeneration passed. The same seven focused
tests passed in 3.97 s. Independent read-only review confirmed unchanged types,
constness and constructor/correlator overloads. Diff --check passed. Logs:
/tmp/xca-dense-array-aliases-{build,test}.log. Source and work log are committed
separately; unrelated user changes remain untouched.

## Dense NDA vector aliases — 2026-11-03

Source: `fe0cc1e` `[dense] use vector aliases`.

Adopted the user's vec_cvt alias in all dense evaluator constructor declarations,
definitions and explicit instantiations. The vec_t and vec_vt aliases remain
available; this class currently has no owning NDA vectors or mutable vector views.
Rank-one array views and std::vector uses retain their existing aliases and types.

Full CPU/Python build passed. The updated clair also regenerated bindings with
unrelated generator-version changes; these were discarded after review confirmed
identical constructor types and Python API. Recompiled the existing tracked
wrapper directly using the _triqs_xca/fast target, then passed all seven focused
CTest entries in 3.92 s (action, three dense tests, Py_api,
Py_one_time_correlator and Py_dense_manual_NCA_OCA). Independent review and
diff --check passed. Logs: /tmp/xca-dense-vector-aliases-{build,bindings-build,test}.log.
Source and work log are committed separately; unrelated user work is preserved.

# Diagram evaluator API and MPI work ownership

Drafted 2026-11-02 on `dev_backbone`, starting at `c4589ac`.
Implementation evidence belongs in `work_log_diagram_index_api_cleanup.md`.
Keep source and documentation in separate, terse commits; do not push.

## Objective

Expose one topology at a time to Python and one prepared Backbone at a time to
C++ numerical tests. Remove linear-index selection from the public evaluator
contract. The evaluator selects this MPI rank's work internally and returns its
local sum. Python accumulates topologies/orders before reducing the completed
sum. Numerical Backbone/BackbonePath execution remains local and MPI-independent.

This replaces the earlier proposal to make interval bounds the primary evaluator
API. Enumeration may initially use integer ranges internally, but callers must
not depend on that ordering or on a particular partition of it.

## Contracts and boundaries

- Python-facing evaluator methods take the propagator, topology, and any required
  observable inputs. Keep ordinary self-energy, dense paired self-energy,
  single-particle GF, and one-time correlator operations. Return a fresh local
  contribution; do not reduce, broadcast, or require a collective in these calls.
- Remove scalar-index, index-vector and interval overloads, public evaluator
  count methods, and index-taking print/native convenience methods. Counts and
  decoding remain with diagram enumeration and its tests. Remove
  FlatIndexSelection after its final consumer is migrated.
- Keep prepared Backbone evaluation and sparse complete BackbonePath evaluation
  available to C++ only. A sparse Backbone call discovers and sums its paths;
  it does not ask the caller to construct them. Retain the existing numerical
  scratch/accumulation semantics in this change, including dense reset() where
  prepared tests need it, but stop wrapping reset() in Python.
- MPI belongs in evaluator orchestration, never Action, Arcs, Backbone, path
  discovery, or numerical contraction routines. Use the installed TRIQS MPI
  library. Stream work; do not materialize all indices/backbones/paths.
- Initially use the world communicator already used by the Python Solver.
  Keep this choice at the orchestration boundary, not scattered through kernels.
  Custom Python communicators require matching evaluator partition and Solver
  reduction, plus binding support; do not add a half-working communicator API.
  C++ serial reference tests use the local prepared interfaces. A future explicit
  communicator option must apply to the complete computation.
- The application owns MPI initialization/finalization. The installed C++ MPI
  library's launcher detection does not initialize MPI; guard orchestration
  against an uninitialized/finalized launched runtime rather than allowing an
  MPI abort. Normal Python package import initializes MPI through TRIQS. Test
  that import route under mpiexec. Python and C++ have differently named forced
  initialization environment flags; do not assume those flags are interchangeable.
  Support ordinary serial execution and the shared launcher-detected world mode.
- Preserve numeric conventions and existing tolerances, not bitwise agreement
  across MPI sizes. Rank-local inputs must describe the same model, propagator,
  topology sequence and observable layout for the final reduction to be meaningful.
  Empty ranks return correctly shaped zeros and still join the final reduction.

## Compact paired self-energy work

The ordinary self-energy has a choice of direction for the special arc. A pair
already evaluates both directions, sharing its internal Backbone computation.
Partition the internal arc choices only: `(2 * n_flavours * n_poles)^(order-1)`
unique pairs, with the usual first-order convention of one pair even when there
are no internal choices. Zero internal choices at higher order give zero work.

Decode pair choices directly into Arcs with the special direction fixed to the
existing direction-zero (backward) convention. Prepare the Backbone and call
the existing paired numerical method once. Do not partition the ordinary
self-energy index range and discard direction-one entries; do not derive a pair
ordinal by scanning or filtering that range. Counting the compact space must not
require the twice-as-large ordinary count to fit in a signed long.

Keep the existing paired algebra, orientation reversal, `T - U`, normalization
and signs. A locally evaluated prepared pair can still start from either endpoint
direction. Test it against the explicit sum of two opposite-direction Backbones.
Partition pair ordinals evenly for arbitrary rank counts, including more ranks
than pairs. This improves distribution of pairs without promising equal cost
for general block-sparse work.

## Python reduction ownership

Replace repeated count/split/interval code with whole-topology evaluator calls.
The internal topology/order loops accumulate local results. Reduce only when a
public Solver operation promises its complete result:

- self-energy across all requested topologies and orders, before Dyson iteration;
- the public paired self-energy order method, after all its topologies;
- single-particle GF across all requested topologies and orders;
- one-time correlator across all requested topologies and orders.

Keep the public Solver single-topology helpers globally complete by reducing
their one local result. Their complete-result contract differs deliberately from
the rank-local C++ evaluator contract. Avoid internal calls through these global
helpers. Sum scalar diagram prefactors locally before the final reduction.
For block GFs a final reduction phase may still involve one reduction per block;
do not introduce packed result buffers in this change.

Audit callers outside solver.py, including the dimer benchmark's private topology
loop call. Move callers requiring a complete topology result to the public Solver
helper. Document that evaluator methods now return local sums under MPI.

## Preserve numerical tests without retaining the public index API

The shared test_utils::Evaluator adapter deliberately exposes historical fixture
indices. Keep that convention inside test code: prepare a Backbone from the
fixture index and call the local numerical API. Vector fixtures sum these local
calls, preserving duplicates, order and empty-selection shape. Full references
also sum local backbones, independently of MPI partitioning. Do not replace
independent manual/closed-form oracles with the production topology loop.

Prepare external observable matrices and statistics with the existing dense and
sparse construction helpers. Keep this data preparation outside diagram code;
do not add another public indexed evaluator overload to avoid test migration.
Keep integration assertions for invalid symbolic observable parity, zero/empty
operators, incompatible atom_diag layouts and malformed propagators on the
production topology APIs, including empty-rank execution. Calling a validation
function only from a test adapter is not evidence that the production API still
validates those inputs.
Use explicit Arcs in new prepared-backbone tests where that improves clarity.
Retain decoding tests and large-count/overflow coverage with the enumeration
helpers. Retain wrong-role, ownership, changed-propagator, parity, empty-data and
strided-topology checks when replacing old public-range tests.

## Reviewable implementation checkpoints

1. **Compact pair enumeration** (`[diag] enumerate self-energy pairs`).
   Add the direct pair count/decoder with independent digit, coverage and overflow
   tests, while existing production adapters remain usable.
2. **Isolate fixture indexing** (`[test] evaluate prepared backbones`).
   Migrate shared and direct C++ numerical consumers to Backbone/BackbonePath
   execution while old production overloads still build. Adapt the benchmark
   consumer, retaining historical measurements with their original labels and
   naming the new prepared measurements accurately. Validate independent references.
3. **Own MPI work in the evaluators** (`[diag] distribute topology work`).
   Add compact pair enumeration and private bounded partitioning using TRIQS MPI.
   Remove evaluator index/count/print overloads and obsolete selection plumbing;
   migrate Python Solver calls and move reductions in the same building commit.
   Regenerate bindings and migrate range/API tests. If reviewability improves,
   compact pair enumeration can be a separate preceding source checkpoint.
4. **Validate distributed execution** (`[test] check distributed diagram sums`).
   Add genuine multi-rank checks for dense/sparse self-energy, pairs, SPGF and
   correlators against local prepared numerical references. Exercise uneven and
   empty assignments and delayed multi-topology reduction. Verify Solver public
   methods retain complete results and do not reduce once per topology.
5. **Adversarial review and completion**.
   Audit numerical signs, pair coverage, MPI initialization, API leakage, topology
   validation before empty execution, generated wrappers and external callers.
   Correct findings and record final evidence and remaining limitations.

Each building source checkpoint gets a separate work-log commit explaining the
motivation, changed call flow and verification. Freeze source edits while builds
are running. Combine dependent changes where an intermediate commit would fail
to build; explain any subdivision in the log.

## Validation and acceptance

- Preserve independent reference values/tolerances and meaningful prepared tests.
- Compare compact pair execution with two explicit endpoint directions and total
  ordinary self-energy; cover several topologies, mixed statistics and pole signs.
- Verify unique pair/ordinary coverage, empty ranks, uneven sizes, first order,
  zero choice spaces, counts above INT_MAX and checked count overflow without
  allocating arrays proportional to the count.
  In particular, order 63 with one flavour/pole on a 64-bit long has 2^62 pairs
  although the ordinary count overflows; sampled decoding must still work.
- Use dedicated MPI tests with MPI initialized explicitly. Existing gtest_main
  executables are serial; launching them under mpiexec does not initialize MPI.
  Include an odd rank count (e.g. three) and more ranks than first-order pairs.
  Register the three-rank CTest with a timeout and PROCESSORS 3. Check rank-local
  ownership independently of the final sum, including exactly one active pair
  rank at first order. Instrument Python reduction calls: the final reduction
  count must not grow with the number of topologies or orders.
- Check numerical equivalence after accumulating multiple topologies and orders,
  plus one final reduction phase in Python. No bitwise MPI-order requirement.
- Configure CPU cbuild only when file/CMake changes require it; source
  ~/env/xca-cpu.sh, build at -j6, run the full CPU/Python CTest suite with
  OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1, parallelism 2. Compile backbone_cpu
  explicitly and inspect regenerated bindings and git diff --check.
- No GPU port, new performance benchmark, topology generator migration or new
  scheduling framework. Future GPU optimization targets high-end cards such as
  GH200, rather than tuning the API to this workstation's RTX A4000.
- Preserve unrelated .gitignore changes, local configure scripts, caches, tools
  and untracked user files. Do not push or rewrite history.

## Future extension

A later order/multiple-topology C++ entry point can distribute and batch work
across topologies. Keep the present one-topology and one-Backbone entry points
usable without forcing a sequential topology schedule or stored global plan.
Equal index counts are only an initial scheduling policy: sparse path costs can
differ substantially. MPI distribution and GPU batching may evolve independently
of these logical input/output contracts.

## Review status

Reviewed independently for MPI/paired scheduling and numerical test migration on
2026-11-02. Revisions before implementation:

- Make MPI runtime ownership explicit and require a safe initialization check;
  distinguish launcher detection from initialization and exercise Python import.
- Preserve production observable/layout validation after moving fixture indexing
  into tests. Keep explicit-Arcs versus decoded-Arcs comparisons independent.
- Retain complete-result Solver helpers, including the dimer benchmark caller;
  keep evaluator calls rank-local and postpone reduction in the accumulation loops.
- Decode pairs directly. A compact pair count must not depend on first forming
  the ordinary count, and there must be no rejected partner work slots.

The plan is ready for implementation. Evidence and any necessary checkpoint
adjustments will be recorded in the separate work log.

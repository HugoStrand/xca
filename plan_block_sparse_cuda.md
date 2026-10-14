# Block-sparse CUDA: investigation and experiment plan

Date: 2026-10-14. Status: implementation in progress. Exact sparse model packing,
CPU path controls and a bounded workload inventory are implemented. Owned-stream
rectangular CUDA products and DLR operations are qualified at the primitive level.
A bounded device reference now covers self-energy, SPGF and one-time correlators
on small physical controls. Consolidated bindings and opt-in Python/MPI solver
integration are implemented and locally qualified. Broader capacity qualification
and the remaining experiments below are still pending.
This replaces the earlier stub. Preserve the dense implementations,
benchmarks and all existing experiments, including negative results.

The performance target is **GH200-class accelerators, not the local RTX A4000**.
Local measurements establish correctness and screen ideas; target measurements must
establish dispatch thresholds, memory policy and production defaults. Here N means
fermionic modes (spin orbitals), not spatial orbitals whose spin doubles N.

## 1. Questions to resolve

The likely useful decomposition is sparse operator application, dense propagator
convolution, and bounded scheduling of independent block paths. Test that hypothesis
before choosing a backend. A single maximum-block-size switch cannot represent all
three costs. The investigation must decide:

- When signed maps, general sparse products, small rectangular kernels or cuBLAS
  should execute each operation, including mixtures within one backbone.
- Which parallelism survives path pruning: DLR nodes, output sectors, indexed
  evaluations, topologies, or combinations of these.
- Which intermediates merit reuse and which must be streamed to fit large sectors.
- Whether topology sharing and CUDA graphs improve complete calls after discovery,
  packing, graph construction, invalidation and retained-memory costs.

Python may change freely to orchestrate MPI. C++ should own local bounded work
queues and accumulation. This plan does not freeze the Python API, introduce
reduced precision, or require an immediate complete 14-mode solver benchmark.

## 2. What the current implementation actually does

Source anchors: [atomic operators](c++/triqs_xca/atom_diag.cpp),
[block operator construction](c++/triqs_xca/block_sparse/atom_diag.cpp),
[block storage](c++/triqs_xca/block_sparse/block_op.cpp),
[evaluator](c++/triqs_xca/block_sparse/diagram_evaluator.cpp),
[public API](c++/triqs_xca/block_sparse/diagram_evaluator.hpp),
[dynamic operators](c++/triqs_xca/block_sparse/dynint.cpp),
[indexing](c++/triqs_xca/backbone.cpp),
[topologies](python/triqs_xca/diag.py), and [Solver](python/triqs_xca/solver.py).

### Basis and structure

The evaluator already uses the **Fock basis within each atom_diag subspace**.
`get_operator_block_impl` converts atom_diag eigenbasis matrices using
`U_target * F_eigen * U_source^dagger`. Thus fundamental c/cdag blocks are signed
partial permutations in the evaluator basis, even when those blocks are large.
The dense storage and round trip conceal this exact structure. Propagators can
still be dense within a subspace; sparse vertices do not make the convolution sparse.

Atom_diag supplies Fock-state lists and fundamental-operator ordering. Construct
operator structure directly from those lists instead of thresholding the rotated
matrices. For mode j and bit state s, annihilation is zero if j is empty; otherwise
it maps to `s xor (1 << j)` with sign `(-1)^popcount(s & ((1 << j)-1))`.
Creation uses the same sign and requires an empty mode. Use atom_diag's actual mode
ordering and state type, with checked shifts, rather than assuming orbital labels
are consecutive bit indices. Build inverse state-to-(subspace,local-index) maps and
verify the resulting edges against atom_diag's c/cdag connections.

For retarded operators, apply the many-body monomials directly to Fock states,
rightmost operator first, using TRIQS's canonical conventions. Investigate reusing
`triqs::hilbert_space::imperative_operator` rather than duplicating fermion algebra.
Coalesce duplicate entries in a specified order; do not prune small coefficients as
an optimization. atom_diag applies a complete monomial in the full Fock space
and then projects its final state; tracing individual projected c/cdag connections
is insufficient for occupation-restricted atom_diag instances. Preserve this policy. Density operators are diagonal, individual hopping/spin-flip terms
are partial maps, and sums may need general CSR/CSC storage. `get_op_mat` alone is
not a direct Fock export: its matrices are in the atomic eigenbasis.

The current dynamic-operator resolver requires one target subspace per source and
an injective connection map for the adjoint; it rejects unsupported multi-target
sums and all-zero operators. Preserve this supported domain. Arbitrary matrices
provided through lower-level APIs also need a general dense/sparse fallback with
explicit structural provenance; never assume all supplied matrices are signed maps.

Symmetry sets group orbitals with identical complete annihilation-connection rows.
The quartet stores Fs/Fdags and pole-weighted barred variants, not just fundamental
operators. Barred operators are linear combinations inside a symmetry set and can
have multiple nonzeros per row/column. Test both sparse materialization per pole
and implicit application of maps plus hybridization coefficients. Respect the
existing cross-set hybridization compatibility check.

The current symmetry-set analysis is not fully optimal: it can separate operators
that could be grouped, including multiorbital density operators with 1-by-1 Hilbert
space blocks. Revisit these grouping restrictions during the investigation.

### Arithmetic, paths and output

| Stage | Current operation | CUDA implication |
| --- | --- | --- |
| Left vertex | F(M,K) times T(r,K,A) | Sparse left apply; anchor A changes at the special vertex |
| Right vertex | U(r,A,K) times F(K,L) | Sparse right apply needs its own layout and transpose treatment |
| Edge | Dense block propagation plus DLR transforms/convolution | Rectangular GEMMs and shared rank-r transforms remain |
| Values to coefficients | Rank-r solve with variable spatial RHS count | Reuse pivoted LU; concatenate or chunk independent RHS |
| Special vertex | T times F_kappa, coefficient mixing, then F_mu times intermediate | Stream or fuse sparse applications; avoid storing every kappa buffer |
| Correlator | Left/right fragments and endpoint contractions | Preserve existing fragment reuse and right reflection |

Flat indices select orbital/pole/direction data; they do not enumerate all subspace
paths. Self-energy additionally loops over output blocks and symmetry sets. Absent
operator connections and zero propagator blocks prune paths, and the special-vertex
symmetry-set constraints prune further. Shape histograms alone therefore cannot
predict ready work or total time.

`set_zero_block_indices` currently uses a whole-block maximum below `1e-16`.
This is a numerical input-dependent mask, not just immutable connectivity. Preserve
its policy initially and invalidate schedules when the mask changes. Cross-set
coefficient tolerance is a separate policy. Test both near their boundaries.

Self-energy accumulation also discards each completed path unless its maximum
absolute entry is greater than `1e-16`. Preserve this additional numerical policy
in the first device reference and test its boundary separately from the propagator
mask. Structural accepted-path counts therefore include candidates that contribute
nothing numerically. Removing or changing this cutoff requires a separately qualified
experiment; batching must not apply it to an aggregate instead of individual paths.

Correlators have a different starting-root rule: their walks test the mask after
internal operators, without first rejecting the root. A two-mode scalar regression
shows that a masked root with stored values around `1e-18` can still contribute to
SPGF when the internal walk reaches live sectors and the endpoint closes onto that
root. Masked BlockDiagOpFun blocks retain their values. Uploading only live blocks
or substituting zero for every masked block would change this behavior. Preserve
the actual starting values and empty-walk semantics in the initial device port.

Core field maps are injective, so each correlator fragment map is injective too.
For either side, `sum_root d_root*d_end <= sum_b d_b^2` by Cauchy–Schwarz. Actual
rectangular side storage is therefore bounded by one all-sector function per side,
rather than `nblocks` maximum-sized functions. This excludes temporary tensors,
transforms, inputs/output and allocator overhead. The descriptor collector checks
this injectivity and rejects stale masks before endpoint visitation.

Different output blocks can execute independently. Contributions to the same output
need explicit ownership and a reproducible accumulation order. Preserve caller index
order and topology order initially. Avoid overlapping cuBLAS batch outputs and
input/output aliasing: CPU expression temporaries cannot be replaced by unsafe device
views. Use live-region storage and ping-pong buffers with explicit lifetimes.

Current block-sparse public functionality comprises ordinary SE, SPGF and one-time
correlators, with full/single/range variants where currently exposed. Explicit
noncontiguous index selections remain available in C++ for experiments. It has
**no paired-SE API**. Both line directions already occur in ordinary indexing.
Paired SE would be a separate extension, not a missing port requirement. CPU Python
evaluation and MPI orchestration now use half-open range bounds; a native block-sparse
port should retain this interface without materializing enormous vectors.

## 3. Evidence collected and scale estimates

### Bounded atom_diag probes already run

CPU probes used a density-density Hamiltonian with unequal onsite energies and
interactions, first with automatic partitioning, then with generic complex hopping
and only total-number Q conserved. These are setup/metadata timings from one run,
not evaluator performance measurements.

| N | Automatic occupation blocks | Q-only block dimensions | Q-only setup + metadata |
| --- | --- | --- | --- |
| 4 | 16 blocks of size 1 | 1,4,6,4,1 | 0.0005 s |
| 6 | 64 blocks of size 1 | 1,6,15,20,15,6,1 | 0.0043 s |
| 8 | 256 blocks of size 1 | 1,8,28,56,70,56,28,8,1 | 0.0939 s |
| 10 | 1024 blocks of size 1 | 1,10,45,120,210,252,210,120,45,10,1 | 2.2578 s |

The occupation fixtures produced N singleton orbital symmetry sets; the Q-only
fixtures produced one set of size N. This changes the special-vertex workload as
well as matrix dimensions. The two Hamiltonians above probe regimes; comparing
their timings is not a same-physics symmetry speedup measurement.

A separate six-mode complex-hopping probe compared direct Fock annihilation maps
against atom_diag eigenbasis matrices rotated back to Fock space. Across all modes
and connected subspaces there were 192 exact nonzeros versus 4752 stored dense
entries. Maximum reconstruction error was `2.11e-15`; structurally zero entries
contained roundoff up to `1.06e-15`. Direct construction removes that roundoff.
High-order numerical equivalence still needs testing; no CUDA result follows from
this probe. Reproduction details are in [the work log](work_log_block_sparse_cuda.md).

### Q-only 10- and 14-mode memory

For Q-only sectors, `d_q = binomial(N,q)`,
`sum d_q^2 = binomial(2N,N)`, and
`sum d_q*d_(q-1) = binomial(2N,N-1)`.
For N=14 the dimensions are
`1,14,91,364,1001,2002,3003,3432,3003,2002,1001,364,91,14,1`.
These arbitrary rectangular dimensions differ from the dense solver's powers of two.

The following are analytical complex-double payload sizes in GiB (2^30 bytes).
G means one all-sector rank-r function. The CPU evaluator has four maximum-sized
rank buffers plus `smax` kappa buffers; here `smax=N`. Its correlator also initially
allocates two maximum-sized arrays per sector before replacing them by actual shapes.

| N | r | One G | One maximum rank buffer | (4+N) rank buffers | Initial correlator side arrays |
| --- | --- | --- | --- | --- | --- |
| 10 | 16 | 0.044 | 0.015 | 0.212 | 0.333 |
| 10 | 26 | 0.072 | 0.025 | 0.344 | 0.541 |
| 10 | 36 | 0.099 | 0.034 | 0.477 | 0.749 |
| 14 | 16 | 9.565 | 2.808 | 50.548 | 84.247 |
| 14 | 26 | 15.542 | 4.563 | 82.141 | 136.902 |
| 14 | 36 | 21.520 | 6.319 | 113.734 | 189.556 |

G, output, transformed copies, library workspace, operator storage and construction
peaks are additional; the correlator column is additional to persistent evaluator
scratch and excludes its extra Tmuop rank buffer and construction temporaries. Do not
copy this allocation strategy to CUDA or batch B copies of it.
In contrast, N=14 occupation blocks need only 6.5 MiB for one G at r=26. Coarsening
symmetry to improve GPU occupancy can catastrophically increase memory and work.

Dense storage of both bare operator orientations costs
`32*N*binomial(2N,N-1)` bytes: 0.050 GiB at N=10 and 15.622 GiB at N=14.
Including both barred families at p=2 poles raises raw quartet payload to 0.150 and
46.866 GiB respectively, excluding copies and atom_diag's own storage.
All N=14 annihilation maps together have only `N*2^(N-1)=114688` nonzeros.
One half-filled c_j block is 3003-by-3432 with 1716 nonzeros.
For a fully mixed barred family, an upper bound per orientation and pole is
`N^2*2^(N-1)` nonzeros; both orientations at p=2 require about 98 MiB of values,
plus indices. Implicit maps plus coefficients may be smaller still.

These numbers make sparse export and lifetime-aware workspace prerequisites for
large-sector feasibility, even before deciding which operator kernel is fastest.
They do not eliminate atom_diag's host-side diagonalization or eigenbasis caches;
measure host peak RSS separately and avoid constructing N=14 atom_diag by default.
The tabulated r=16/26/36 and p=2 are capacity examples, not promises of adequate
resolution for the target physics. Determine actual rank and pole count from beta,
energy window and accuracy on smaller physical models, then repeat admission estimates
with the intended values. Include rank/pole sensitivity before claiming N=14 feasibility.

## 4. Representative workloads and bounded sampling

Maintain three families with fixed seeds, explicit Hamiltonian/operator definitions,
partition metadata and independently checked topology/index lists:

1. **Many tiny blocks:** diagonal hybridization, density-density interactions and
   automatic occupation partitioning, N=4/6/8/10/14. Nonidentical energies and time
   dependence prevent a constant-scalar shortcut. Include diagonal retarded density
   interactions. Scalar vertices still require nontrivial DLR work.
2. **Mixed blocks:** physically valid automatic or explicitly conserved partitions
   from spin/orbital models, with small arbitrary dimensions and adjacent rectangular
   shapes. Include noncommuting complex propagators, supported spin-flip/retarded
   operators and alternation between small and large operations in one path.
   Record actual partitions; do not assume a model name fixes block sizes.
3. **Q-only:** generic number-conserving complex hopping and interactions, first
   N=4/6/8, then bounded N=10; use the exact N=14 sector shapes in isolated replays.
   Include dense compatible hybridization, since diagonal coefficients alone would
   understate the single large symmetry set's mixing cost.

On small systems, also compare finer and coarser valid partitions of the **same**
Hamiltonian and hybridization against a dense reference. Do not discard incompatible
hybridization elements to make a desired partition legal. Separate synthetic dense
matrix replays from complete physical evaluator measurements.

For order m, current ordinary flat-index count is `2^m*(n*p)^(m-1)`;
correlators use `2^(m-1)*(n*p)^(m-1)`. Here n includes applicable dynamic channels.
Subspace paths add work beyond these counts. With n=14,p=2,m=12 the ordinary count
is 33970214778764787712, beyond signed 64-bit indexing. This must produce a checked
rejection unless a separately designed hierarchical index API is introduced.

Topology generation is another limit: `diag.py` visits pairings before filtering;
raw pairing counts are `(2m-1)!!` (654729075 at m=10). Use a small explicit validated
catalogue at high orders, including different crossing/nesting patterns and sharing
opportunities. Do not generate all topologies and then select a few. Select fixed
valid index ranges or explicit indices without scanning the whole domain. Report
both proposed indices and accepted subspace paths so survivor-only sampling cannot
masquerade as Solver throughput.

| Tier | Work allowed | What it establishes |
| --- | --- | --- |
| T0 | Analytic N=14 inventory; actual atom_diag metadata through N=10 | Shapes, connectivity and memory bounds, no large diagonalization |
| T1 | Exact large rectangular primitive shapes; 1/2/4 DLR-node GEMM batches; narrow spatial RHS slabs with all r nodes | Large-shape throughput and memory, not a complete evaluator |
| T2 | Complete N=4/6/8 calls; bounded low-order exhaustive controls and selected high-order topologies/indices | Path/sign correctness and scheduling behavior |
| T3 | N=10 few-topology/few-index full calls; N=14 selected-sector/path diagnostics only after a memory check | Larger-path behavior with explicit coverage limits |
| T4 | Complete target-hardware calls only after measured time/memory admission | Production qualification for the tested scope |

Screen each candidate with one pilot and up to three samples, normally at most ten
seconds per case and a 15-minute campaign budget. Reduce case counts rather than
silently changing shapes when a pilot is expensive. Finalists get at least three
fresh-process repetitions and enough samples to resolve variance. Record timeout or
memory rejection as a result, not a missing row. Full N=14 diagonalization/evaluation
requires a separate measured resource estimate; primitive success does not qualify it.

## 5. Experiments, controls and decisions

### E0 — Work and memory inventory

Instrument CPU path discovery with bounded descriptor traces. Record `(M,L,K)`, r,
operator type/nnz, symmetry-set size, block/anchor IDs, direction, topology/index,
accepted/rejected paths, output fan-in and dependency edges. Weight histograms by
executed work and bytes as well as counts. Separate atom_diag/export, path discovery,
operator application, DLR solves/transforms, dense convolution and reduction time.
A null arithmetic scheduler measures descriptor generation, pruning and bucketing.

Deliver an admission estimator including host/device live peaks, input/output,
transforms, sparse indices, handles/workspaces and caches. Reject over-budget runs
before allocations. Validate predictions against measured peaks at small sizes.
Decision: choose only a few representative sequences and identify whether scalar
scheduling, operator bandwidth, convolution or capacity dominates each family.

### E1 — Direct sparse export and a numerical reference

Build direct fundamental maps and supported many-body sparse operators from
atom_diag Fock metadata. Compare compact signed maps, CSR/CSC and dense fallback.
Validate subspace ordering, fermionic signs, adjoints, anticommutation, complex
coefficients, repeated-term cancellation and current unsupported-operator errors.
Permute fundamental-operator labels/order and subspace state order in dedicated checks;
a test using only consecutive labels would not detect a mode-to-bit convention error.
Build barred operators from maps and coefficients, not from rotated dense arrays.

Use three comparisons: existing CPU with rotated operators; CPU with direct-Fock
operators; CUDA with exactly the same direct-Fock inputs. This separates basis-roundtrip
roundoff from device errors. Check full diagram outputs at existing strict tolerances,
including cancellation-sensitive high-order samples. If old/new CPU disagree beyond
those tolerances, investigate with small independent/manual references before changing
acceptance criteria. Keep the production path unchanged until this is resolved.

Implementation observation (2026-10-14): one added order-4 total-number sample
differs between exact-Fock and rotated-operator CPU results by `5.83e-25` on a
`5.49e-12` block signal (relative `1.063e-13`). Replaying the rotated quartet through
the CPU evaluator reproduces the old result bitwise, isolating the changed operator
inputs; the direct signed-map/CAR controls and manual NCA check qualify the exact
fields. Existing order-1/2/3 CPU comparisons retain `1e-13`; newly sampled order-4/5
old-versus-new CPU comparisons characterize this amplification with a `1e-12` bound.
CUDA-versus-exact-input CPU keeps `1e-13` through the sampled fifth order. This is
limited qualification, not a general high-order cancellation guarantee or a reason
to change the current production CPU path.

Correlator observation (2026-10-14): an added order-4 bosonic sample differs by
`1.186e-21` between CUDA and exact-input CPU on a `1.712e-9` signal. Independent
long-double scalar arithmetic using the same cached double LU/rank/kernel constants
finds CPU error about `4.8e-13` relative and CUDA error about `2.3e-13`. A CPU-order
trace probe does not explain it; single-RHS solves reduce but do not eliminate it.
Both probes were reverted. Order-1/2/3 device controls retain `1e-13`; added order-4
bosonic controls use `1e-12` against both CPU and the arithmetic control. This control
shares qualified connectivity descriptors, so it isolates arithmetic precision;
it does not independently validate traversal or DLR approximation error. Larger
orders/shapes and stronger cancellation still require qualification.

Measure export cost, host RSS, bytes uploaded and changed-coefficient update cost.
Decision: which structural representation is reliable and compact; kernel speed is
not the only reason to retain exact sparse construction.

### E2 — Minimal allocation-bounded CUDA reference

Establish straightforward device execution of already discovered paths using actual
rectangular live buffers, general library primitives and explicit sparse/dense operator
representations. Reuse validated dense-native stream, handle, upload-event and LU
ownership principles, without importing square or maximum-dimension allocations.
Keep a small dense-operator reference on small cases for cross-checking sparse applies.
Own cuBLAS/cuSOLVER/cuSPARSE resources; preserve a clear synchronous API boundary.
Do not silently replay submitted work through another evaluator after device errors.

This reference must cover both traversals, special vertices and each current observable
before broad fusion. Its purpose is an auditable correctness/performance control;
large N=14 execution is not required to establish it.

The initial self-energy reference uses one cached CSR/plain-transpose pair and
recomputes kappa products inside the mu loop to stream the special vertex in CPU
summation order. Its exact per-path cutoff uses a bounded host scan of device output.
These are auditable reference choices, not selected performance strategies. Explicit
storage admission includes input/output, actual rectangular work buffers, rank
tables/RHS, kernel weights and a configurable sparse-workspace reserve; a global
free-memory check adds headroom. Sampled free-memory deltas include other processes,
so compare them with predictions only in isolated runs. Complete host/opaque-library
peak qualification and larger shapes remain E0/E2 work.

The initial correlator reference retains actual rectangular fragments (one function
capacity per side), executes the reverse traversal and reflection on device, and
forms sparse endpoint products followed by nonconjugating device traces. It uploads
all stored starting-block values, including masked tiny inputs. Endpoint snapshots
can be temporary, so their address-based operator cache is cleared on entry/exit,
including exceptions. Index-local results receive parity before ordered global
addition; no self-energy magnitude cutoff is imposed on correlators. The conservative
correlator admission bound includes a third function capacity, two output tensors
and trace storage. Its large-sector bound needs schedule-sensitive refinement before
claiming practical N=14 execution on target hardware.

Integration checkpoint (2026-10-14): `Solver(..., block_sparse_backend='cuda')`
selects the reference with automatic or explicit atomic symmetries; CPU remains
the default. The consolidated CUDA module exposes real/complex atom_diag factories,
retarded models and range calls. Device and sparse-workspace budgets are solver
preferences preserved in archives. GPU setup, index-count queries and range
evaluation are collectively guarded before MPI reductions. Two-rank tests cover
numerical results plus rank-local setup/count/compute failures and recovery.
Library-only CUDA builds work with native CUDA-small disabled. This integration
does not select optimized dispatch or establish target-hardware throughput.

Complete-call checkpoint (2026-10-14): the shared bounded benchmark now records
unchanged CPU and CUDA-reference calls for all observables and ordinary/retarded
models. The [local baseline](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/README.md)
contains 30 successful processes and 82 raw samples, including a zero-survival
six-mode occupation window. The eight-mode number case is approximately at parity
(245 ms CPU / 265 ms CUDA). A single measured ten-mode number-sector evaluation
(maximum dimension 252) takes 7.01 s CPU / 1.14 s CUDA, whereas ten-mode scalar
sectors take 1.02 s CPU / 5.60 s CUDA. These are first-index crossing windows,
not full-sum or target-hardware results; ten-mode cases have only one repetition.
That baseline ratio preceded the E5 algorithm-matched CPU control below; use
the subsequent matched complete-call comparison when assessing device advantage. Small-sector work should prioritize launch,
packing and DLR amortization rather than selecting sparse products from these
complete-call timings alone. Detailed phase/launch profiling is still pending.

### E3 — Operator applications: maps, sparse libraries, dense products

For left and right products compare:

- Direct signed gather/scatter without atomics where a partial permutation permits it.
- Implicit sums of maps weighted by pole/hybridization coefficients.
- Materialized barred CSR/CSC and general retarded-operator sparse products.
- Dense complex-double GEMM, including small rectangular kernels as controls.

Sweep row degree, sparsity, reuse count, RHS width and layout on traced shapes;
include the exact 3003-by-3432 case. Account for format conversion, transpose storage,
preprocessing, packing and coefficient updates. Dense-right-sparse application may
need a transposed formulation; distinguish plain transpose from conjugate transpose.
For reductions over multiple entries, define and test accumulation order explicitly.

cuSPARSE SpMM supports complex double, but deterministic algorithm, layout and batching
capabilities differ. In particular CSR_ALG3 is non-transposed CSR, disallows conjugate
transpose of B and does not support batched SpMM. Compare unbatched deterministic CSR,
pretransposed storage, supported COO alternatives and custom ordered kernels rather
than assuming one API supplies all requirements. Verify the installed version and
workspace costs; preprocessing state belongs to its descriptor and workspace.
[cuSPARSE documentation](https://docs.nvidia.com/cuda/cusparse/index.html)

Decision: a small capability table keyed by structure/shape/reuse, with reproducible
numerical behavior. Do not infer complex-double acceleration from real or reduced-
precision tensor-core peak rates.

E3 checkpoint (2026-10-14): [operator_apply](experiments/block_sparse_cuda/operator_apply/README.md)
compares a one-launch, ordered general CSR kernel against the retained deterministic
cuSPARSE path and a dense cuBLAS strided batch with shared operator storage. The
fixtures use exact adjacent-number Fock sectors without diagonalization, individual
signed maps and complex weighted field sums. Both multiplication directions use
plain-transpose CSR for right products, including empty rows. It is not yet a compact
map representation or implicit weighted sum.

The [local sweep](experiments/block_sparse_cuda/operator_apply/thinkstationP620_RTXA4000/README.md)
passed all 24 processes / 432 rows with exact replay; native output matched the CPU
sparse control bit for bit and both library controls had relative error below
5.34e-16. A width-17 tail/multi-entry memory check reported zero errors. The sweep
includes the exact 3003/3432 N=14 geometry with bounded RHS width 1/16, not full
sector-sized rank functions or N=14 diagonalization. It records nnz/row degrees,
packing/setup/first-use and warm timings, explicit operator/workspace bytes and
source/binary/library/toolchain provenance. At width 16 for all-field sums, ordered
CSR / retained cuSPARSE left application takes 0.0056/0.333 ms at 4 modes,
0.0369/0.337 ms at 10 modes and 0.572/0.786 ms at 14 modes; right applications show
similar benefits. The cuSPARSE reference also charges per-node descriptor queries
and synchronization, so these ratios do not establish fundamental library limits.

Next test the native route in an isolated complete evaluator, with the existing
reference retained. Qualify budgets, lifecycle, strict numerics and inclusive
complete calls before promotion. Width/degree/reuse-dependent dispatch, wider RHS,
compact maps, implicit sums, COO alternatives and target hardware remain pending;
no production dispatch threshold is selected from these primitive measurements.

### E4 — Where to batch and where to fuse

Compare one path at a time with bounded batches across output sectors, flat indices
and topology cohorts. Within a cohort, compare ready-operation shape buckets against
fused compatible path segments. Include homogeneous, heterogeneous, sparse-survivor
and tail workloads. Charge the null scheduler, descriptor upload, gather/scatter and
ordered fan-in to the complete segment time. A modest batch-size sweep is sufficient;
avoid a Cartesian product of every tuning parameter.

The checked local cuBLAS headers expose no C/Z grouped-GEMM entry point, and the
current GroupedBatchedEx type table does not include complex double. Use uniform-shape
ZgemmBatched/StridedBatched buckets, individual GEMMs or custom complex kernels as
actual controls; recheck capability on the target installation. Do not design around
hypothetical heterogeneous complex grouped GEMM.
[cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex)

Use bounded windows and ordered output ownership so preserving contribution order
does not require retaining every lane's result. For huge blocks reduce active lanes
before exhausting memory. Decide per operation/segment, not by global maximum size.
For tiny arbitrary dimensions test a small tile family or runtime dimensions before
compiling an unbounded collection of rectangular specializations.

### E5 — DLR transforms, solves and convolution

Reuse pivoted rank-r LU and test packing independent spatial RHS across blocks/lanes.
Keep all r nodes for each transformed element; spatial slabs are independent but
arbitrary time-node slicing is not. Compare packing/solve/unpacking with separate
solves, including final short slabs. Preserve complex double and reject the previously
unsuccessful custom-solve approach unless new independent numerical evidence warrants it.

The installed cppdlr rank-3 convolution recomputes whole Hilbert transforms inside
the time-node loop. Establish an algorithm-matched CPU control with those transforms
hoisted, as well as the unchanged CPU baseline. Otherwise a GPU gain could largely be
an algorithmic CPU improvement. Cache transformed/weighted G only with complete keys:
block, rank/mesh, weights/poles, direction and propagator version.

For scalar blocks, compare fusion across DLR work and independent paths; sparse
vertices alone may be negligible. For large blocks, profile dense convolution after
sparse vertices have removed their share of work. Decision: which transformations
and resident coefficient copies earn their bandwidth and memory cost.

E5 checkpoint (2026-10-14): [dlr_hoisted](experiments/block_sparse_cuda/dlr_hoisted/README.md)
is an isolated CPU control that hoists the two complete-array Hilbert transforms
without changing products/addition order. Eight primitive geometries/ranks (108
samples) match unchanged CPU bit for bit, for coefficient convolution and an edge
including both coefficient solves. The largest bounded edge (210/252/210, rank 23)
takes 894/210/56.5 ms for unchanged CPU / hoisted CPU / retained CUDA, respectively;
CUDA includes download, with inputs/scratch resident. A private generated evaluator
uses exactly the current CPU path/operator/accumulation code and a cached rank-ops
adapter, without changing production sources or installed headers. Generation checks
the two convolution sites and regenerates on source changes.

The [clean full-call control](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/hoisted-cpu-02/README.md)
contains all 15 original physical workloads (30 processes, 82 samples). All hoisted
CPU outputs match unchanged CPU bit for bit, including retarded/reverse traversal
and zero-survival controls; CUDA screening checks pass. Eight-mode number sectors
now favor hoisted CPU (131 ms versus 267 ms CUDA). The ten-mode number sample still
favors CUDA (5.04 s versus 1.13 s); scalar sectors favor hoisted CPU (0.795 s versus
5.61 s CUDA). Ten-mode rows have one repetition, not a noise-qualified promotion.
The preceding hoisted-cpu-01 timings are excluded because they overlapped a CPU
verification build; numerical results and the exclusion are retained. Target runs,
threaded CPU comparison, changed-G replay, RHS grouping/caching and detailed phase
profiling remain pending. No dispatch choice follows from hoisting alone.

### E6 — Special-vertex mixing

Compare the current kappa-buffer strategy with streaming kappa/mu tiles and fused
sparse applications. Test singleton sets, one N-wide set, diagonal and dense compatible
hybridization, both line directions and bosonic/fermionic endpoints. The backward
case transposes coefficient indices and uses reflected hybridization; preserve this.
Only skip coefficients that are structurally known to be zero under the current policy.

Reassociation or precontraction of the double sum changes rounding and may change
memory demand. Validate it separately from launch fusion. Decision: a bounded algorithm
whose measured traffic/time improves the complete special-vertex segment, without
N simultaneous maximum-sized intermediates.

### E7 — Large-sector memory and placement

Test liveness-based reuse, output spatial slabs, streamed special-vertex buffers and
bounded caches before increasing batch size. Include actual N=14 shapes in primitive
replays without creating all atom_diag matrices. If a convolution requires K tiling,
account for changed reduction order and traffic; this is not equivalent to independent
RHS chunking. Report cold/warm input/output and coefficient-copy peaks separately.

On the target, measure available HBM, host/Grace memory bandwidth, copy/managed-memory
behavior and NUMA placement rather than treating all memory as equivalent. Compare
explicit staged slabs against fully resident execution where both fit. Graph-retained
buffers count against the same budget. Leave configurable headroom for Solver/MPI.
Account for the complete call's promised output: SE returns all output sectors, whereas
SPGF/correlators have different endpoint outputs and scratch. A selected-sector diagnostic
is not a drop-in replacement for a full API call. Include output transfer/host storage in
the streaming estimate and specify what happens if even the required output cannot fit.

Distributing indices over several GPUs still replicates model data. If replication
cannot fit, sector/tile ownership needs a distinct communication experiment with
input exchange and load balancing; equal numbers of sectors do not balance half-filled
work. Do not present multi-GPU index partitioning as a solution to per-GPU capacity.
Decision: a demonstrated memory envelope and a rejected/accepted full-call admission
policy; retain streaming even when it sacrifices peak GEMM throughput to make a case fit.

### E8 — Topologies, shared subcomputations and graphs

Start with independent topology cohorts. Measure reusable prefixes/suffixes on actual
operator/block paths, including low-overlap controls. Existing correlator fragment
reuse is part of the baseline. A candidate shared node must agree in all numerical
inputs, weights, block paths, parity placement, layout and versions, not merely have
a similar topology drawing. Cache structural schedules separately from numerical values.
Only share identical intermediate computations initially. Factoring a common term out
of the topology sum changes arithmetic grouping and is a separate numerical experiment,
even if the resulting algebra is formally equivalent. Bound DAG discovery itself;
never enumerate the full topology/index space merely to search for reuse.

Compare four cases on the same bounded workload: independent execution; independent
execution captured in graphs; an explicit shared-computation DAG; that DAG in graphs.
Graph replay does not memoize values. Changed G or coefficients must recompute numerical
nodes, and changed numerical zero masks can invalidate the structure. Include topology
cohort changes, final tails, graph construction/update, cache misses and retained bytes.
[CUDA graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

Capture only supported operations with stable lifetimes; keep path discovery and
forbidden synchronization outside capture. Evaluate break-even replay count from
setup cost divided by per-call saving after updates; no positive saving means no
amortization. Decision: retain graphs or sharing only if measured complete-call gains
survive realistic input changes and memory pressure. Neither is a prerequisite for
an initial operational evaluator.

### E9 — Full functionality and MPI integration

Port existing C++ observable tests to an evaluator seam without replacing the CPU
reference or copying its implementation into a fake CUDA facade. Cover ordinary SE,
SPGF, one-time correlators, dynamic interactions, right reflection, parity, empty and
zero paths, unequal rectangular blocks, changed G/coefficients/masks, unsupported
operators and overflow. Test full calls, individual indices and bounded ranges against
explicit-vector sums, including empty ranges, tails and noncontiguous caller order.
Keep numerical tolerances from the existing observable tests; add repeatability and
cancellation-heavy cases. Compare small cases to manual/dense references as applicable.

Exercise stream/instance lifetime, asynchronous upload, buffer reuse, allocation
failure before submission, device failure after submission and sanitizer checks.
Capability rejection must occur before work submission; submitted partial results
must not be duplicated by fallback. CPU-only builds must remain supported.

Python owns MPI assignment, device placement and collectives. Native code owns local
chunks; ranks may have unequal queues or no work. Coordinate errors before reductions
and keep collectives outside variable local loops. Measure topology/index hybrid load
balance and reduction/transfer costs. Changing global summation grouping needs explicit
numerical validation. Test Solver integration and serialization without fixing Python's
current per-topology call pattern as an optimization constraint.

## 6. Benchmark organization and acceptance

Proposed new shared fixtures belong in `benchmarks/block_sparse_common`, with CPU
and CUDA entry points in `block_sparse_cpu` and `block_sparse_cuda`. Generic methodology
and definitions stay at folder roots; machine metadata/results go in informative
hardware subdirectories. Keep experiments isolated under `experiments/block_sparse_cuda`
and retain all existing dense experiments. Reuse recorder concepts from
[the dense common benchmark](benchmarks/dense_common/README.md), not square-only fixtures.
Enable targets through the existing benchmark CMake option independently of examples.

Record commit/toolchain/library versions, CPU/GPU/memory, partition and operator
fingerprints, r/p/order, explicit topology/index selections, survivor counts, numerical
errors, source hashes, warm-up and repetition policy. Separate:

- Model construction and operator export; first allocation/upload/compilation/capture.
- Warm fixed-input replay; changed-G calls; changed coefficients or topology cohorts.
- Device segment time, synchronized C++ API time and Python/MPI complete-call time.
- Host/device peak bytes, allocations, launches, copies and cache hit rates.

Compare unchanged CPU, algorithm-matched CPU, minimal device reference and each
candidate on identical inputs and semantic coverage. Include low-survival and
zero-survival index windows alongside
selected active paths; disclose selection rules and avoid deriving full-sum speedups
from an active-only sample. Fix CPU thread counts and also
report a sensible threaded CPU control for large matrices. Kernel-only rates inform
mechanisms; dispatch decisions use packing and scheduling inclusive segment/call time.
Profile separately so profiler overhead does not contaminate timing rows.

Predeclare promotion criteria after measuring noise: provisionally require a repeatable
15% complete-segment improvement on its intended regime, no unexplained correctness
change and no more than 5% regression on protected cases unless dispatch excludes them.
Report absolute latency too. These are engineering screening thresholds, not evidence
that a whole Solver improved; final decisions require full-call and target-hardware
measurements. Prefer a few explainable routes over an unstable large tuning table.

## 7. Execution order and deliverables

1. E0 and E1: inventory, reproducible physical fixtures, exact operator export,
   numerical qualification and checked resource estimates. Stop if operator-domain
   semantics or old/new CPU agreement remain unresolved.
2. E2: a complete small-system device reference for current block-sparse observables.
3. E3/E5/E6: investigate the two dominant regimes separately: many tiny blocks and
   large Q-only sectors. Prioritize sparse representation and large-case memory
   feasibility; establish the hoisted CPU convolution control.
4. E4/E7: add bounded batching and memory policy using observed ready-work sequences.
5. E8 only when traces justify topology sharing or launch amortization. It must not
   delay a useful simple backend. E9 correctness runs throughout, MPI tests as the
   native interface becomes operational.
6. Repeat finalists on GH200-class hardware, publish measured capability/crossover
   tables and revise the implementation plan before committing to production dispatch.

The operational small-system reference and its local qualification are checkpoints
toward the implementation goal. Optimized dispatch, target-hardware qualification
and a 14-mode production workload remain unqualified.

## 8. Adversarial review

This was a separate adversarial **self-review** of the first committed draft, checked
against source and the probe recipes. It was not an independent model review.

| Challenge to the first draft | Resolution or required gate |
| --- | --- |
| Small r and p examples could make the large system appear feasible without adequate physical resolution. | Label them capacity examples; derive intended beta/window/accuracy ranks and poles, then repeat admission and sensitivity checks. |
| The correlator memory table could be read as a complete peak estimate. | State that Tmuop, construction temporaries, G, outputs and library workspaces are additional; E0 must measure the full live peak. |
| Streaming a selected output sector could quietly weaken the public API. | Separate sector diagnostics from full SE/SPGF/correlator coverage and include full output/transfer storage in admission. |
| A direct-Fock test with consecutive orbital labels could agree despite a general mode-order bug. | Add permuted fundamental labels/order, state ordering, adjoint and algebra tests before sparse export becomes authoritative. |
| Active-path-only high-order sampling could greatly overstate Solver throughput. | Require low/zero-survival windows and disclose sampling; no full-sum extrapolation from active-only timing. |
| Prefix reuse might really factor the topology sum and change cancellation behavior. | Share identical intermediates first; classify algebraic reassociation as a separate numerical experiment and bound reuse discovery. |
| Sparse-library speed could rely on nondeterministic batching unavailable in the deterministic mode. | E3 capability-tests algorithms/layouts and charges alternative packing/transpose costs; repeatability is an acceptance gate. |
| Avoiding dense operator copies might still leave atom_diag host initialization prohibitively large. | Measure host construction/RSS independently; N=14 metadata/shape replays do not require or qualify a complete atom_diag instance. |
| Large speedups might come from correcting CPU convolution structure rather than GPU execution. | Keep unchanged and transform-hoisted CPU controls on identical workloads. |
| Dense-small success on A4000 could bias dispatch for different hardware and rectangular shapes. | Require target-hardware qualification and per-operation measured policies; preserve the simple device reference. |

Outstanding experimental decisions: sparse-map versus CSR/CSC crossover, viable
large-sector liveness policy at actual r/p, surviving path parallelism, numerical
acceptance of direct-Fock export, and whether topology sharing/graphs amortize.
The plan intentionally leaves these to measurements rather than selecting winners.

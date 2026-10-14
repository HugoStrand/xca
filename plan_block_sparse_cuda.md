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

### Arrhenius experiment budget

The user's Arrhenius allocation is **100 GPU-hours/month**, shared with their other
work. Treat this as an allocation ceiling, not a budget to consume for this project.
Screen implementations, correctness and workload selection locally first. Use one-GPU
GH200 jobs with explicit short walltime limits, reuse compiled Arm build trees, and
retain results so failed or completed cases are not repeated unnecessarily.

Start with a focused build/smoke job capped at ten minutes (one sixth of a GPU-hour
at one allocated GPU). Then select a few representative many-small, mixed-block and
large-sector cases from local evidence, with a declared cap of one GPU-hour for the
initial benchmark cohort. Expand only when those results identify a specific unresolved
question. Avoid full size/order/topology Cartesian sweeps and full N=14 atom_diag
construction as initial target experiments; use bounded sector/operator probes.

Record each job ID, allocated GPU count, elapsed time and accounting charge where
available, including build time and failed jobs. Check the allocation's remaining
balance before further campaigns and leave room for the user's other workloads.
Queued jobs and scheduler validation do not constitute hardware qualification.

First target smoke evidence: Arrhenius job 3818566 built pinned source 3f462c3
with GCC 14.3.0/CUDA 12.9.86 and native block-sparse sm_90 kernels. Four focused
C++ CTests (20 GoogleTests) passed in 10.63 seconds. The one-GPU allocation lasted
120 seconds, nominally 0.0333 GPU-hours. Evidence is retained under
`experiments/block_sparse_cuda/gh200_qualification/Arrhenius_GH200/smoke-01`.
Python/MPI, full-suite, target performance and large-system gates remain open.

Initial performance evidence: job 3818907 reused that Arm build and completed
seven matched CPU/library-CUDA/ordered-CUDA cases (63 samples) in 91 seconds of
one-GPU allocation, nominally 0.02528 GPU-hours. Numerical screens and exact
repeatability passed. Ordered CUDA beats the unchanged CPU on sampled N=8/N=10
number-sector NCA calls (1.46x/28.47x), but loses on scalar/mixed small cases and
N=6 retarded SPGF. One scalar window rejects every path; retain it as overhead
evidence, alongside its active neighboring window. Rank 23, one topology/index
and three samples do not establish a dispatch crossover or full Solver speedup.
Evidence: `benchmarks/block_sparse_common/Arrhenius_GH200/initial-01`.
Next target experiment: retained scalar-path batching with semantic gates and
active/rejected windows. Initial target allocation total is 211 seconds, nominally
0.05861 GPU-hours. Broader Python/MPI, capacity and target policies remain open.

Scalar target evidence: job 3819263 ran the retained constants/arena/fan-in candidate
against ordered GPU, unchanged CPU and matched batched CPU. Fourteen active semantic
tests and 72 samples passed, with matching structural counts. The eight-index scalar
range and N=10 scalar case reduce GPU latency by 108.7x/191.2x versus ordered GPU,
but the matched CPU remains faster on the range; N=10 GPU has only a 1.045x advantage
over matched CPU in this small sample. General correlator fallback is unchanged.
The order-six chain window has zero surviving paths, so active high-order/cutoff
qualification remains required before promotion. Evidence is under
`benchmarks/block_sparse_common/Arrhenius_GH200/scalar-01`. Keep the candidate
experimental while closing those gates; no dispatch policy follows from this cohort.
This job used 90 seconds, nominally 0.025 GPU-hours; total target allocation across
the three jobs is 301 seconds, nominally 0.08361 GPU-hours.

Local active-order-six gate: the scalar experiment now discovers an active chain
index in a bounded metadata search (index 10 for the two-mode/one-zero-pole fixture).
Ordinary-amplitude contributions, 24 cutoff-bisection probes, margin-separated
threshold sides and separately allocated changed-input reuse pass across the four
scalar variants. Initial 1e-13 comparison failed; independent long-double arithmetic
finds reference/candidate errors about 4.28e-13/3.90e-13. Only this new order-six
case uses a 1e-12 oracle bound; older comparisons and production policy are unchanged.
Focused memcheck reports zero errors/leaks. This shares qualified connectivity and
cached double constants, so it isolates arithmetic rather than DLR approximation.
ULP-adjacent high-order cutoff behavior, broader poles/topologies/scalar sets and
target rerun of the new control remain gates before promotion.

Adjacent-cutoff finding: the active order-six fixture has representable amplitudes
around 0.10775810972137 where batched constants discard a lane retained by general
CUDA. Preserve that baseline as a known semantic-failing candidate. A separate
`experiments/block_sparse_cuda/scalar_cutoff` variant replays completed lanes within
an experimental 1e-26 band around the 1e-16 cutoff through general CUDA arithmetic,
then applies the existing ordered fan-in. The adjacent probe and full local suite
pass; replay-owned input copy/lane are charged to admission and focused memcheck
reports zero errors/leaks. A five-sample ordinary-range screen shows no evident
regression. The band is not a proven bound for arbitrary conditioning/order or
cancellation. Broader stress, replay cost/capacity and GH200 qualification remain
required; production is unchanged.

Cutoff-cohort gate: replay input storage is now copied once per cohort and reused
by sensitive lanes. A private observer verifies actual replay across 512 lanes and
a tail and per-call reset. New order-two tests cover seven threshold-neighbor
amplitudes with zero and nonzero pole choices, comparing counts/output/repeatability
against ordered general CUDA. All 17 local replay controls pass; focused cohort
memcheck reports zero errors/leaks and preserved constants controls pass afterward.
Evidence is in `scalar_cutoff/thinkstationP620_RTXA4000/cohorts-01` under experiments.
This strengthens bounded replay ownership/semantics, not the fixed band's universal
validity or a production/target dispatch policy.

Multi-field scalar checkpoint: separate `experiments/block_sparse_cuda/scalar_sets`
candidate admits all-scalar propagators with multiple operators per symmetry set.
It batches independent lanes while preserving outer mu/inner kappa sums, supports
mixed set widths and reserves three additional batch buffers. Singleton cohorts
retain the earlier operation sequence; correlators/non-scalar shapes remain general.
Eighteen local controls pass, including complex retarded two-/three-field sets,
NCA/OCA ranges, changed owners/masks, repeatability and admission. Targeted memcheck
is clean; preserved singleton and CPU-only recipe controls pass. Evidence is under
that experiment's `thinkstationP620_RTXA4000/sets-01`. Dedicated multi-set timing,
GH200 qualification and conservative cutoff policy remain before promotion.

Explicit multi-set timing checkpoint: three dedicated frontends validate scalar
dimensions and grouped shifted-density fields and share complex retarded coefficients
with their numerical controls. A bounded local N=4/6/8 campaign passes all 36 samples,
matching GPU path counts and exact replay. Batched OCA latency is 9.36–10.17x below
unchanged CPU and 60.12–89.63x below ordered general CUDA; small NCA still favors CPU.
These are complete calls on an A4000 proxy, not algorithm-matched CPU comparisons
or whole topology sums. CPU-only/default-fixture compatibility checks pass. Evidence:
`benchmarks/block_sparse_common/thinkstationP620_RTXA4000/density-sets-01`.
Next add a matched batched multi-set CPU control before attributing hardware gains,
then qualify a small GH200 cohort and settle the cutoff/promotion policy.

Matched multi-set CPU checkpoint: host control now shares scalar eligibility,
512-lane discovery/cohorts, ordered special recipe, packed LU/transforms and fan-in
with the GPU candidate. Ordinary-case screening/replay pass in 48 local samples;
matched CPU outputs equal original CPU exactly on this cohort. GPU versus matched
CPU is about even at N=4 OCA, 1.09x faster at N=6 and 0.89x at N=8; small NCA
favors CPU. Thus most earlier gains are algorithmic, not established GPU hardware
advantage. CPU-only and focused GPU multi-set/adjacent-cutoff controls pass. Host
control does not implement the GPU uncertainty-band replay, so arbitrary cutoff
equivalence remains separate. Evidence: common local `density-matched-01` campaign.
Next qualify this bounded four-route cohort and new semantic controls on GH200,
within the existing resource cap, before deciding production promotion/defaults.

GH200 multi-set checkpoint: job 3823469 on pinned source 6987c4d passes five CTests
(38 active controls, including new high-order/cutoff/cohort/multi-set cases) and all
48 matched benchmark samples. GPU/matched-CPU latency ratios expressed as CPU/GPU
are 0.25/0.60/0.92/0.82 for N4 NCA/N4 OCA/N6 OCA/N8 OCA: matched CPU wins every
sampled case. GPU greatly improves on general CUDA, but no target hardware win
over matched host batching is established. Evidence is in the common GH200
`density-matched-01` campaign. Prioritize scoped profiling and reduction of host
descriptor/weight preparation, uploads and synchronizations before promotion.
Investigate whether GPU-side preparation or fewer/fused launches change this result.
Do not introduce an automatic dispatch policy from these few cases. The fixed replay
band, wider numerical regimes, Python/MPI and larger-capacity gates remain open.

This one-GPU job used 446 seconds, nominal 0.12389 GPU-hours; target total is now
747 seconds (0.2075 GPU-hours). Archive timestamps ahead of the cluster clock caused
rebuilds; source metadata was normalized without content edits and the recipe now
prevents recurrence. The source snapshot/build and all previous evidence are retained.

Scoped local profiling checkpoint: a warmed N6 OCA multi-set call spans 25.890ms
of CUDA runtime trace, with 2.452ms summed kernel intervals and ~.682ms memory
operations. It makes 272 stream synchronizations, 156 allocations, 115 host-to-device
copies and many upload-event barriers. Wrapper/export time is not benchmark data;
API/device intervals can overlap and tracing affects latency. Evidence is retained
in `experiments/block_sparse_cuda/scalar_sets/thinkstationP620_RTXA4000/profile-01`.
Source inspection identifies field assignment/parity recomputed inside the DLR-node
weight loop although constant per lane/operator pair. Next isolate its hoisting and
compare unchanged GPU/matched CPU inclusive calls; then consider resident immutable
weight tables and device descriptor packing to reduce uploads/synchronizations.
Preserve the current candidate/profile as controls and repeat semantic/cutoff gates
before target timing or promotion. No GH200 inference follows from this local trace.

Parity-hoisting checkpoint: separate scalar_signs candidate prepares field indices
and parity once per lane/operator pair, preserving weight arithmetic, sums and
cutoff/fan-in. Matched host control shares that preparation; baselines remain.
All18 controls and48 old/new CPU/GPU samples pass, plus CPU-only exact screening.
Local GPU OCA speedups are1.55–1.89x; matched CPU1.55–1.83x. New GPU wins sampled
N6 over matchedCPU by1.18x, is near even atN4 OCA, and loses elsewhere. GPU summary
norms/errors/counts match old/new, without claiming cross-route bitwise equality.
Evidence is common local density-signs-01. Device payload/admission are unchanged.
Next reduce repeated immutable weight uploads and stream/event barriers through a
separate resident-table/device-descriptor candidate, preserving current timing and
numerical controls. GH200 qualification and conservative promotion remain required.

### 2026-10-14 — Resident target qualification and reusable staging

GH200 resident qualification job3828369 completed0:0, oneGPU245s (.06806GPU-hours).
FiveCTest entries/39controls and36matched samples pass. Resident gains12–28% over
parityGPU, but matchedCPU wins all four bounded cases. Target dispatch must follow
complete-call evidence, not localA4000 crossover. Total tracked allocation .27556
GPU-hours; retain100GPU-hours/month constraint and defer broad sweeps.

Separate scalar_staging reuses cohort-owned weight/constant/pointer buffers with
owned-stream copies; completion still precedes source rewrites and destruction.
NDA nontrivial descriptor extent-constructor host fault fixed with ordered initial
host-vector upload. Nineteen local tests and three targeted memchecks pass;36local
samples show6–9% gain over resident,12,288extra device bytes charged to admission.
No target staging qualification, API-count profile or production promotion yet.
Next trace staging, then isolate barrier reduction with source-lifetime checks.
Cutoff band proof/promotion, broader capacity/Python/MPI and integration remain open.
Evidence: common/CPU/CUDA hardware folders density-staging-01 (local) and
 density-resident-01 (GH200); all older variants retained.

### 2026-10-14 — Upload events and next production gates

Separate scalar_streams preserves staging while allowing queued kernels to proceed
after uploads. Host-source reuse waits on the prior copy event; device reads/writes
share one stream. Explicit recipe copy-back and destructor completion remain. Full20
local controls, three focused memchecks and36samples pass. Local complete-call gains
4–7% over staging with unchanged admitted array storage. Scoped trace streamSync
278→185 plus87eventSync; transfers/kernel count unchanged. Both remain experimental.

Keep the fixed cutoff replay band's unproven envelope explicit. Next prioritize
bounded capacity qualification and production numerical/promotion decisions; avoid
endless micro-variants given modest local gains and matchedCPU target results. Reuse
these candidates for one bounded GH200 finalist comparison only when it resolves a
production decision. Target usage remains .27556GPU-hours; no new allocation this step.
Evidence: local density-streams-01 and staging/streams profile-01. All baselines kept.

### 2026-10-14 — Production payload preflight and rank sensitivity

Host-only reference_memory_plan now shares generic storage arithmetic with production.
CPU-only binomial N10/N14/r26/r64 tests and live CUDA policy/observable bounds agree
exactly. Retained generator seams adapted with old storage contracts preserved;
71 focused CUDA controls and two CPU metadata controls pass. Metadata CLI evidence
is in common/CPU local reference-capacity-01, not a full14-mode or GH200 allocation.

Host-sector Reference N14 requires the current admission envelope52.081/74.898GiB
for NCA/convolution at rank26, versus128.154/184.318GiB at rank64. Rank64 exceeds a
120GiB payload budget before extra context/host construction overhead; rank26 is a
capacity assumption only. Never substitute assumed low rank for physical resolution.
Planner flags compare conservative payload bounds, not actual free-memory admission.

Next investigate a tighter NCA live-tensor bound: source shows six special-vertex
rank tensors including T, versus seven generic reserved tiles. Qualify all special
fallbacks, workspace lifetime, reject/accept boundaries and measured peak before
using that reduction. This may help NCA; convolution paths need their own capacity
strategy. Keep full output/API and caller input storage accounted. Remaining large
allocation, optimized promotion and Python/MPI/full-suite gates stay open. Target
budget unchanged .27556GPU-hours; no new cluster allocation for metadata work.

## 1. Questions to resolve

Resident weight checkpoint: separate scalar_resident candidate reuses pole device
vectors and constructor-owned forward/reflected hybridization tables. Bounded lane
descriptors replace rank-sized weight uploads, retaining multiplication/sum order.
Persistent payload enters fixed bounds for every observable; descriptors enter
scalar admission. 18 inherited controls plus an exact expanded-weight selector
test pass, and three focused memcheck controls report zero errors/leaks. All36 local
matched samples pass. GPU speedups over parity baseline are1.21–1.32x; sampled OCA
beats matchedCPU, but small NCA does not and N8's3% margin is weak evidence. Added
explicit bytes48,352/71,904/101,344 at N4/N6/N8. Common local density-resident-01
retains numerical/storage/timing evidence. Scoped trace shows host-to-device bytes
7,883,316→1,014,324 (7.77x), still115 copies,156/158 alloc/free and272 stream syncs.
Earlier trace predates parity hoisting, so total span reduction is not residency-only.
Next owned reusable staging/descriptor buffers and fewer barriers, with source
lifetimes/stream ordering qualified before target timing or promotion. The fixed
cutoff replay band and broader integration/capacity gates remain.

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

E3 complete-call candidate checkpoint (2026-10-14): a private generated
`OrderedDiagramEvaluator` changes only the sparse snapshot type in the current
reference source. Its bounded snapshot owns CSR/plain transpose and the device
owner, checks payload before upload, rejects aliases/layout/device errors and
finishes applications before caller temporaries expire. The reference remains
unchanged. The candidate passed all eight inherited evaluator tests plus two
snapshot budget/ownership/zero-entry tests. Four focused memcheck cases reported
zero errors and zero bytes leaked; local one-device execution does not qualify
cross-device switching.

[Ordered full calls](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/ordered-evaluator-01/README.md)
passed all 30 processes / 82 samples with unchanged discovery counts, CPU screening
and exact replay. All original observables and ordinary/retarded fixture regimes
are included. Eight-mode number sectors take 85.7 ms candidate / 266.8 ms retained
GPU (prior clean run) / 135.2 ms matched CPU. Ten-mode number sectors take
690.8 / 1130.5 / 5064.8 ms, with only one repetition there. Ten-mode scalar sectors
still favor CPU (2.71 s candidate versus 0.788 s CPU), so E4/E5 amortization remains
necessary. The zero-survival window is retained. The candidate's highest screening
relative error is 3.65e-13 in a higher-order weak contribution; strict inherited
numerical test gates were unchanged.

The complete-call evidence supports qualifying an optional native operator route
in production next, retaining the library path for control/no-native-compiler
builds. Before promotion, wire explicit capability/backend selection through the
consolidated bindings and Solver, and run full CPU/CUDA/library-only/MPI tests.
Do not select a target default or claim all-small-block speedup from these local
selected-index results. Wider RHS, changed-operator costs, peak capacity, batching
and GH200 measurements remain open.

Production C++ checkpoint (2026-10-14): `SparseBackend::Ordered` is an
explicit optional route on `DeviceSparseBlock` and the block-sparse CUDA evaluator;
`Library` stays the default. `ordered_sparse_available()` reports compiled support.
`XCA_CUDA_BLOCK_SPARSE=AUTO|ON|OFF` controls native kernel compilation independently
of dense CUDA-small. AUTO leaves the library route usable with CUDA NDA but no
CUDA compiler, and CPU builds omit native kernels. The installed/exported static
kernel target uses ordered complex-double arithmetic with FMA contraction disabled.

This uses the reference's owned CSR/plain-transpose snapshots, budgets, stream/device
checks, overlap checks and cleanup, with one rank-wide native launch and synchronous
return. The same eight evaluator tests pass for both production routes, capability/
budget tests pass, and focused endpoint/cache memcheck reports no errors/leaks.
A CUDA build with both native families OFF passes library/capability tests and
rejects ordered requests; CPU guarded compilation passes. The original private
experiment still builds and passes after constructor adaptation. Python bindings,
Solver/archive selection, full rebuilt suites, MPI and production-path timings are
the next qualification step; the private candidate's timings are not substituted
for a measured production implementation. No target default has been chosen.

Python/MPI checkpoint (2026-10-14): the consolidated facade accepts
`operator_backend='library'|'ordered'`, and Solver persists
`block_sparse_operator_backend` (library default). Capability checks in gpu.py and
the facade reject unsupported native selection before model export/device setup.
Older archives restore library defaults; CPU evaluation retains preferences without
loading CUDA. All observables, ordinary/retarded partitions, self-consistency,
rank-local count/setup/compute failures and recovery are tested for both routes.
Complete suites pass: CPU 77/77, CUDA 87/87, native-disabled CUDA 77/77. Two-rank MPI
passes all five Python tests on each rank. The operator selector is now integrated,
not only an isolated candidate; library and all experiments remain available.

[Production timing](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/ordered-production-01/README.md)
measures the actual descriptor-owning production route in 30 isolated processes /
82 samples, with counts matching reference and numerical/replay gates passing.
The eight-mode number window takes 92.9 ms native / 266.8 ms library (prior clean
run) / 130.2 ms matched CPU. Ten-mode number sectors take 698 / 1130 / 5091 ms,
while scalar ten-mode sectors take 2846 ms native versus 802 ms CPU. Ten-mode rows
have only one repetition. The remaining small-block bottleneck therefore requires
E4/E5 work; this does not select a GH200 default or prove complete Solver speedup.
The production snapshot retains descriptor creation even for native application,
so its cost differs from the private snapshot as recorded.

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

E4/E5 scalar-edge checkpoint (2026-10-14): production rectangular products
already batch across DLR nodes. [scalar_edges](experiments/block_sparse_cuda/scalar_edges/README.md)
instead packs independent scalar edges across spatial RHS columns, using the same
cached LU/getrs and shared rank transforms with elementwise scalar products (no
cross-lane outer product). Inputs retain separate owned buffers; inclusive timing
charges per-lane device gather/scatter. Each lane is compared against CPU separately,
including zeros and tiny inputs, and repeated outputs must match exactly.

The [local sweep](experiments/block_sparse_cuda/scalar_edges/thinkstationP620_RTXA4000/README.md)
passed 15 cases / 135 rows across 1–256 lanes and three mesh scales, with maximum
per-lane relative error 2.77e-15 and zero memory/leak errors in a tail-size check.
At rank 23, 32 edges take 0.723 ms packed inclusive versus 9.68 ms independent;
127 take 1.52 versus 38.58 ms; 256 take 2.63 versus 77.67 ms. These are GPU-segment
ratios, not CPU or Solver speedups. Initial upload/setup and host output downloads
are excluded; CPU controls are numerical, with timed matched CPU packing/solve
still pending. No scheduler, cutoff or topology reassociation is implemented.

Next establish timed matched CPU batch controls, then build a bounded evaluator
candidate retaining copied path metadata, per-path magnitude decisions and original
accumulation order. Special-vertex mu/kappa grouping and irregular correlator
fragments need explicit scheduling coverage; do not assume packed scalar convolution
alone handles their semantics. Ready-operation grouping, tails/changed-G and peak
memory must enter full-call evidence before promotion. GH200 measurements remain
required for defaults.

Matched scalar control checkpoint (2026-10-14): timed CPU independent and
packed routes plus GPU pointer-table gather/scatter are now in
[scalar-cpu-01](experiments/block_sparse_cuda/scalar_edges/thinkstationP620_RTXA4000/scalar-cpu-01/README.md).
All 17 cases / 408 rows pass per-lane CPU gates and exact replay; maximum relative
error remains 2.77e-15. CPU packing is essential context: at rank 23 / 127 lanes it
takes 0.202 ms versus 0.471 ms for GPU kernel packing (the per-lane-copy GPU variant
takes 1.55 ms). Larger bounded cohorts favor GPU in this local segment: 512 lanes
0.852 ms CPU versus 0.489 ms GPU; 1024 lanes 1.666 versus 0.763 ms. The 512/1024
points do not establish a hardware-independent threshold or a feasible ready-work
population in a diagram. Other ranks/cohort sizes remain in the table.

Pointer tables borrow still-owned input/output buffers; post-timing in-place input
changes are re-gathered and checked against fresh CPU controls (17/rank32 and
1024/rank23 checks passed). Memcheck reports no errors/leaks. Setup/table upload
and changing buffer addresses are not included in timing. Before a scheduler is
promoted, charge descriptor refresh/discovery and preserve per-path decisions and
original fan-in order. Build a bounded private full-evaluator candidate next,
retaining general native/library routes for unsupported operation cohorts. Do not
read the first GPU-only ratios as CPU speedups; CPU batch comparison corrected that
interpretation before production changes. GH200 remains the target.

Scalar scheduler checkpoint (2026-10-14):
[scalar_paths](experiments/block_sparse_cuda/scalar_paths/README.md) is a private
complete-API candidate with a bounded self-energy specialization for all-1x1 blocks
and singleton symmetry sets. It copies owned Backbones/path descriptors into at
most 512 lanes, separates special-vertex positions, uses packed scalar edges, and
retains the original per-lane magnitude decision and ordered fan-in. Other sets,
block geometries and correlators execute the unchanged general evaluator path.
The API is fully retained; correlator/irregular-set batching itself is not qualified.

Eight inherited tests pass, plus complete-call/memory checks on 640 paths across
cohorts/tails and 2304 paths at ten modes. The corrected
[scalar-path-02 campaign](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-path-02/README.md)
passes 30 processes / 82 rows with exact replay and matching discovery counts.
Ten-mode scalar SE takes 90.7 ms versus prior production native 2846 ms and paired
hoisted CPU 798 ms. A separate three-repetition check measured 89.2 ms median.
Those compare the existing CPU evaluator, not a full batched scalar CPU scheduler;
matched batch-CPU full-call control is still required before attributing all gains
to GPU hardware. The campaign's ten-mode point has one timed repetition.

A failed final workspace-bound rebuild initially left the old executable in a
started campaign; scalar-path-01 is excluded with explicit source/binary mismatch
notes. The corrected build/suite and multi-cohort memcheck pass, and both admission
and the public storage diagnostic include extra batch workspace. This remains a
conservative explicit bound, not complete host/opaque peak qualification. Before
promotion: threshold-adjacent contribution tests, larger symmetry-set sums,
correlator batching, lifecycle/budget/changed-address gates, matched CPU full calls
and GH200 evidence. Retain all general/reference/experimental paths.

Scheduler qualification checkpoint (2026-10-14): four additional independent
[scalar-path tests](experiments/block_sparse_cuda/scalar_paths/qualification.cpp)
now pass alongside the eight inherited cases. They cover more than 512 accepted
paths with a non-full tail and split-range additivity; in-place complex G changes,
changed masks and all-zero inputs; adjacent NCA completed-path cutoff values;
extra workspace rejection before submission and a zero-budget-admission empty
range; and multi-field scalar symmetry sets retaining general execution. Focused
memcheck/leak-check passes all four cases with zero errors/leaked bytes.

Cutoff probes use actual reconstructed hybridization factors rather than assuming
an exactly representable analytic half, and keep input masks live while crossing
the completed-output threshold. NCA has no convolution, so this establishes policy
and accumulation behavior, not every higher-order rounding-boundary case. Matched
batched-CPU complete calls, higher-order threshold sensitivity, multi-field/correlator
batching and production/Python/MPI promotion remain pending. No benchmark or target
dispatch claim changes from these added tests.

Matched complete-path checkpoint (2026-10-14): the scalar CPU control now
shares recipe.inc with the GPU scheduler, using identical eligibility/cohorts,
operation order, cutoff and fan-in but CPU packed cached-LU/rank transforms.
It builds/runs in CPU-only and CUDA trees. Extraction leaves all twelve GPU
qualification tests passing. The
[paired full-call comparison](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-matched-01/README.md)
passes all 30 processes / 82 rows, with CPU outputs bitwise equal to original CPU
and exact GPU replay. Other geometries/observables forward to unchanged CPU/general
GPU, so those are not matched batching comparisons.

The selected ten-mode scalar SE window takes 23.3 ms matched CPU versus 89.4 ms
GPU; four-mode scalar SE takes 2.72 versus 3.43 ms. The zero-survival six-mode
window is retained (0.032 versus 1.98 ms). Thus earlier comparisons to hoisted CPU
mostly captured algorithmic improvement. Do not promote this GPU scheduler based
on old-baseline ratios. Next isolate/reduce serialized per-path cutoff/fan-in
handoffs with output ownership that preserves per-output contribution order, then
repeat the matched complete-call and semantic gates. GH200 remains target hardware;
these timings do not impose a CPU fallback or target default.

Ordered fan-in checkpoint (2026-10-14): the private scalar scheduler now has
an isolated output-owned accumulation variant while retaining the serial control.
The host per-lane cutoff is unchanged. Surviving IDs are grouped by output in their
original lane order; one GPU thread owns each output/rank cell and sequentially
adds to the existing value, without atomics/tree reduction. All preparation/upload
costs are charged. Baseline/variant pass 12/13 tests; the extra cancellation case
checks interleaved owners, existing sums and sequential order. Five focused memory/
leak cases pass with zero errors/leaked bytes.

[Matched full calls](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-fanin-01/README.md)
pass 30 processes / 82 rows with matching counts and exact replay. Ten-mode scalar
SE drops from prior serial GPU 89.4 ms to 56.1 ms, but matched CPU is 22.9 ms.
This does not meet a CPU-beating scalar promotion claim. Target GH200 measurements
and other regimes remain necessary. Next profile remaining host weights/descriptor
preparation, rank-solve/transform synchronization and cutoff transfer, then test
bounded caching/fusion only with complete invalidation and matched controls.
Retain every previous candidate/reference and do not hide the GPU-losing result.

Profile-guided storage checkpoint (2026-10-14): scoped Nsight capture of one
warmed scalar SE call diagnosed thousands of small allocation/copy/synchronization
operations. [Trace evidence](experiments/block_sparse_cuda/scalar_paths/thinkstationP620_RTXA4000/profile-01/README.md)
is separate from timing; profiler wrapper duration includes capture/export effects
and is excluded. The isolated arena variant keeps scalar input/output in two owned
contiguous device tensors, with borrowed block views valid only inside owner lifetime.
Live masks, recipe, cutoff, ordered accumulation and general fallback remain unchanged.
Packing/upload/download/result construction all enter complete calls.

Repeated scoped capture confirms cudaMalloc calls drop 3206→137, cudaFree 3208→139,
cudaMemcpy2D 2055→8 and stream synchronization 1253→230. Both control/arena pass thirteen
semantic tests; five focused memory/leak cases pass with zero errors/leaked bytes.
[Unprofiled paired calls](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-arena-01/README.md)
pass 30 processes / 82 rows with matching counts and exact replay. Ten-mode scalar
SE takes 25.3 ms arena versus prior fan-in 56.1 ms and matched CPU 23.0 ms. Separate
three-sample smoke measures 25.5 ms. Near parity is not a robust matched-CPU/target
promotion win, and ten-mode campaign rows still have one sample.

Next investigate residual host weight/descriptor preparation and synchronized rank
operations using the reduced trace, retaining all controls. Cache/fuse only with
complete model/mesh/topology/propagator invalidation and bounded ownership/admission.
No production/Python/MPI scalar selector, target default or full N=14 execution is
claimed; GH200 remains the target.

Path-constant preparation checkpoint (2026-10-14): the private scalar scheduler
now has an isolated constant-coefficient variant and an algorithm-matched CPU
control. Vertex/endpoint coefficients, prefactors and signs are computed once per
lane. GPU uploads one coefficient per lane and broadcasts it across DLR nodes;
node-dependent pole/hybridization weights retain their full representation. Each
complex multiplication and its position in the recipe are preserved; no numerical
cache or product reassociation is introduced. Prior variants remain available.

Fourteen variant tests and thirteen retained arena tests pass. Six focused memcheck
cases report zero errors/leaks, including exact broadcast-versus-expanded device
products, tails and changed coefficients. CPU-only multi-cohort qualification and
all fifteen paired CPU cases match the original CPU evaluator bit for bit. The
[paired campaign](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-constants-01/README.md)
retains 30 processes / 82 rows and a separate fresh five-sample scalar comparison.
Ten-mode medians: retained GPU arena 24.898 ms, new GPU 21.179 ms, retained CPU
22.398 ms, matched new CPU 20.993 ms. GPU improvement is about 14.9%; matched CPU
also improves, and local GPU/CPU remain essentially at parity. Do not promote or
claim a target crossover from this result. GH200 remains the target.

Remaining scalar gates include higher-order cutoff-adjacent paths, optimized
non-singleton special sets/correlators, broader memory admission and target-hardware
validation. Next distinguish residual rank-operation synchronization from host
path work, and investigate E6 multi-field streaming rather than inferring broad
block-sparse gains from one scalar selection. Production/library defaults and
Python/MPI interfaces are unchanged.

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

E6 bounded-retention checkpoint (2026-10-14):
[special_vertex](experiments/block_sparse_cuda/special_vertex/README.md) retains
private two-slot and sixteen-slot kappa caches. For a q-field symmetry set and c
cached products, right applications drop from q*q to c+q*(q-c), while all weighted
additions and left applications retain their original order. Products are owned
within one special vertex; no cross-call numerical cache or reassociation occurs.
Singleton sets allocate no cache. Admission adds a checked, capped worst-case
rank-buffer envelope before input upload, including the conservative extra
correlator allowance even though correlators do not use the cache.

Both eleven-test suites pass in ordinary and native-disabled library-only CUDA.
Exact-input reference comparisons cover rectangular four-field paths, both sparse
backends, changed G/masks and non-singleton bosonic mixing with complex off-diagonal
coefficients and reflected line directions. Pre-submission rejection, empty ranges
and recovery are checked. Four initial focused cases and the added bosonic case
pass memcheck with zero errors/leaks. CPU-only configuration excludes the fixtures.
Production C++/bindings/MPI execution is unchanged.

The [isolated GPU campaign](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/kappa-01/README.md)
records 21 processes / 105 rows, five samples per process, with unchanged discovery
counts, norms and control-error metadata. Full-set retention reduces selected
number-symmetry latencies by 16–37%; partial retention generally gives smaller gains.
The singleton control regresses about 1%. These are GPU-route comparisons; the CPU
reference already retains kappas, and its control timing is not a matched benchmark.
Timing preceded separate additional bosonic/library qualification without overlap.

Memory limits rule out unlimited retention as a universal policy. Analytic binomial
N=14/r=26 envelopes, without atom_diag/evaluation, add 9.127 GiB for two slots or
63.888 GiB for the full fourteen-field set to the reference's 90.409 GiB variable
envelope. Fixed tables/operator reserve and host/opaque allocations are additional.
These are conservative maximum-square allowances, not observed live peaks. No
N=14 production capability or GH200 dispatch follows from this A4000 evidence.

Next compare a mapped-kappa route against these controls: when each kappa column
has at most one nonzero entry, use an owned source-index/value map to form each
weighted mu buffer directly, then retain the original left sparse application and
mu accumulation order. This can remove materialized q kappa products while charging
bounded maps/weights against admission. Capability checks must inspect actual CSR
structure, including empty columns and complex values; general non-mapping operators
keep a qualified CSR route. Preserve coefficient/reflection/parity placement and
prove numerical behavior before any promotion, including cutoff-sensitive cases.
Continue E7 streaming/admission work and target-hardware qualification; GH200-class
hardware remains the optimization target.

Mapped-kappa checkpoint (2026-10-14):
[mapped_special](experiments/block_sparse_cuda/mapped_special/README.md) implements
the planned sparse-column route. It classifies actual CSR blocks, retaining exact
complex/tiny values and empty columns. A thread forms one mu element by traversing
fields in original order, applying the mapped right product and corresponding
weight, then adding sequentially. The original left application and mu-to-output
addition remain separate. Non-mapping operators/singleton sets use the reference.
There is no q-tensor cache, cross-call numerical reuse or sum factoring. Checked
admission adds only maps and one rank-by-field weight table; all other conservative
function/rank-buffer envelopes remain. Shape checks and stream ownership are explicit.

The fifteen-test evaluator suite and two host cases pass. Seven focused cases pass
memcheck with zero errors/leaks, including complex rectangular kernels, cancellation
order, changed inputs/weights, metadata rejection and a nine-point four-field NCA
cutoff neighborhood. Unsupported structure is witnessed on accepted live paths;
reference fallback is checked. The classifier passes in CPU-only and native-disabled
library-only builds, which omit the compiler-dependent kernel/evaluator. Original
physics tolerances are unchanged; fusion can alter rounding versus library calls,
so this is not a universal bitwise-equivalence or higher-order boundary guarantee.

The [isolated four-route comparison](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/kappa-mapped-01/README.md)
records 28 processes / 140 samples. Selected number-symmetry calls improve 43–78%
over the ordered GPU reference; mixed improves 18%; singleton regresses 1.6%.
Discovery counts agree and maximum mapper screening relative error is 8.567e-15.
Five samples per process are not independent process repetitions. These are GPU
route comparisons, not matched CPU, Solver/topology-sum or GH200 speedups.

Analytic extra map allowances are 64,160 bytes at N10/r23 and 1,158,976 bytes
(1.105 MiB) at N14/r26, versus the prior 63.888 GiB full-set kappa cache allowance.
No atom_diag/evaluation was performed at fourteen modes. The reference's 90.409 GiB
variable envelope and additional fixed/operator/host/opaque costs remain, so the
mapper alone does not prove a fitting fourteen-mode full call. Next replay actual
fourteen-mode Fock maps on bounded row slabs, then refine E7 liveness/streaming and
qualify optional production/Python/MPI integration. Higher-order cutoff-adjacent
paths and target hardware remain gates. Preserve all current controls and reference
fallbacks. GH200-class hardware remains the optimization target.

Production mapped-policy checkpoint (2026-10-14): the qualified special vertex
is now available through ``SpecialBackend::Mapped`` and Solver
``block_sparse_special_backend='mapped'``. ``'reference'`` remains the default;
selection is independent of the library/ordered sparse operator backend. Native
build support is inherited from XCA_CUDA_BLOCK_SPARSE, with explicit unsupported
rejection. The consolidated CUDA module exports compiled capability through
``block_sparse_mapped_available()``. CPU preferences remain stored without importing
CUDA; archives preserve the setting and older archives default to reference.

The production implementation owns maps/weights until stream completion, keeps
original field/left-application order, uses actual CSR eligibility and retains
device-reference execution for non-mapping operators/singletons. Structural
fallback occurs before mapped device work, never after a submitted failure. The
checked metadata allowance is included in reported/admitted storage, including
conservative correlator admission. No global handle or tolerance changes.

Full CPU 79/79, CUDA 96/96 and native-disabled library-only CUDA 81/81 suites pass.
Eight production controls pass memcheck with zero errors/leaks. Python tests cover
both special policies crossed with available sparse backends, all current
observables, dynamic interactions, archives and collective setup/count/evaluation
errors. Two MPI ranks pass five methods each. Concurrent CUDA-tree verification
caused contention; these wall times are correctness evidence, not performance.
Shared policy declarations/capability definitions live outside generated experiment
copies so all retained controls still build.

This completes optional mapped integration, not the whole implementation goal.
Next refine E7 schedule-sensitive admission and scratch lifetimes, especially NCA
versus convolution-bearing paths. Preserve full API output requirements and reject
before submission when the declared working set cannot fit. Higher-order boundary
coverage, broader batching and GH200/full fourteen-mode validation remain gates;
existing experiments and reference defaults remain available.

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

E7 actual-sector primitive checkpoint (2026-10-14):
[bounded Fock slices](experiments/block_sparse_cuda/mapped_special/thinkstationP620_RTXA4000/fock-slabs-01/README.md)
replay one mu row slice with all canonical creation/annihilation maps between
3003/3432-dimensional fourteen-mode sectors. No atom_diag, Hamiltonian or complete
evaluation is built. Independent host checks verify known four-mode signs and
fourteen-mode occupancy/complement counts; each field has 1716 entries. Both
orientations and 32/128 rows at rank26 are exercised, with smaller controls.

Eight isolated processes / 96 samples pass strict controls and exact mapped replay.
The matched serial CPU and mapped GPU outputs agree exactly in this finite suite;
this is not a universal bitwise claim or tuned/threaded CPU ceiling. CPU-only and
CUDA host cases pass, and the fourteen-mode inverse slice passes memcheck with no
errors/leaks. Resident GPU reference retains all snapshots outside timing; a
separate route charges fresh maps/uploads and mu download. Input/output buffers
remain resident, and explicit tensor/map/operator payload is capped at one GiB;
host/opaque memory is not a complete peak measurement.

Fourteen-mode/rank26/128-row resident mapper times are 20.281/17.784 ms for
annihilation/creation, versus separate GPU 43.813/38.943 ms. Map rebuilding plus
host materialization costs 111.836/96.341 ms. This supports keeping mu on the device
through its consumers; slices omit other rows, left application, sum over mu, cutoff,
DLR work, full output and MPI. They qualify actual large-sector indexing and a
bounded primitive, not a fitting fourteen-mode full solver. GH200 remains the target.

Next integrate the mapped route as an explicitly selected production option while
retaining reference defaults and all controls. Requalify CPU-only, CUDA/library-only,
all current observables, archive/error behavior and MPI before enabling it publicly.
Continue E7 scheduling/admission and higher-order boundary work; full N=14 and GH200
dispatch remain unqualified. Do not use primitive CPU ratios as whole-solver evidence.

E7 compact-DLR checkpoint (2026-10-14):
[compact_dlr](experiments/block_sparse_cuda/compact_dlr/README.md) isolates a liveness
candidate. It releases the completed solve RHS before convolution and uses disjoint
caller output for the diagonal contribution; aliases retain a separate diagonal.
Operation/reduction order is unchanged. Checked allocation rejection precedes
release or output writes. Four DLR controls and eight full-observable controls pass;
primitive memcheck reports zero errors/leaks. Production operations/defaults remain.

The [primitive comparison](experiments/block_sparse_cuda/compact_dlr/thinkstationP620_RTXA4000/compact-01/README.md)
requires exact cached/compact outputs across six cases / 36 rows. Square rank18/D256
retention falls by 36 MiB after convolution, but the next solve brings boundary peak
saving to only 18 MiB. Retained and phase-peak storage must remain distinct; aliases
and opaque/caller memory are additional. The private full evaluator reserves five
instead of six DLR tiles because its convolution outputs are distinct.

The [complete-call screen](benchmarks/block_sparse_common/thinkstationP620_RTXA4000/compact-full-01/README.md)
passes 35 processes / 175 samples, with identical discovery/norm/error metadata for
mapped and compact variants. Latency changes are small (-3.9% to +1.4%), not a robust
speedup claim. NCA uses no convolution, so its reduced reported bound is only reserve
refinement. The N14/r26 global envelope loses one 4.563 GiB maximum tile, still without
full-call admission proof. Next separate schedule-specific NCA versus convolution
working sets, then qualify a production memory policy with actual allocation/lifetime
and numerical controls. GH200 remains the target; do not promote from retained-byte
savings alone. Higher-order boundaries, full output/host/opaque peaks and target
validation remain gates.

E7 NCA-admission checkpoint (2026-10-14):
[nca_admission](experiments/block_sparse_cuda/nca_admission/README.md) qualifies a
private topology-aware self-energy bound. The generic accessor stays unchanged;
validated one-line topologies subtract six inactive DLR scratch tiles. Full input/
output, seven caller tiles, fixed tables, sparse reserve, weights and mapped metadata
remain. Higher orders/correlators retain generic admission. Explicit-budget rejection
precedes state changes; eligible NCA calls release completed retained RHS/convolution
scratch before querying available memory, preserving immutable tables and solver.

Ten tests pass, including eight current observable/API controls, exact production
comparisons at the lower NCA budget, one-byte-below rejection before path work, empty
ranges, higher-order rejection/recovery and OCA/NCA/OCA transitions. Independent
release/reconstruction is exact and release idempotent. Two focused memcheck cases
report zero errors/leaks. No production/Python behavior or runtime claim changed.

The analytic N14/r26 variable envelope falls from 90.409 to 63.029 GiB without
atom_diag/evaluation, omitting only inactive scratch. Additional fixed/operator/
metadata/host/opaque costs still apply; no full fourteen-mode fitting proof follows.
Next promote topology-aware accessors/release into production, update budget controls
against the actual schedule, preserve private scalar extra reserves and requalify
consolidated wrappers/MPI. Keep the generic upper estimate available. GH200 remains
the target and full-call/target/higher-order boundary gates remain.

E7 production NCA-admission checkpoint (2026-10-14): the evaluator and consolidated
facade now expose ``self_energy_storage_bound(topology)`` while retaining the generic
no-argument estimate. Validated NCA uses the lower explicit envelope, checks the
budget before state changes, releases completed DLR scratch and then queries free
memory/headroom before input upload. DlrOps release preserves tables/handles and is
idempotent; higher orders/correlators keep their conservative policies. Scalar
experiment generation explicitly adds its extra workspace to admission and both
bound forms. Retained kappa/mapped budget tests now use the actual topology estimate.

CPU 80/80 and library-only CUDA 85/85 full suites pass. CUDA covers all 100 entries:
98 passed in the full run, two stale generic-bound rejection tests were corrected
and passed their focused rerun. Production admission memcheck passes both controls
with zero errors/leaks; six Python methods pass on each of two MPI ranks. Tests
check lower-budget acceptance, higher-order rejection/recovery and OCA/NCA/OCA
transitions for reference/mapped policies. No arithmetic/default change or benchmark
speedup claim. The discarded overlapping build attempt was restarted serially;
all evidence comes from completed final-source verification.

This makes the NCA reservation usable through public C++/Python APIs. Full-function
input/output requirements and host/opaque peak qualification remain. Next inspect
remaining E4/E7 batching and scratch-policy selection against actual supported calls,
and audit conditional E8 graph/sharing evidence before declaring implementation
complete. Higher-order cutoff boundaries and GH200/full fourteen-mode qualification
remain explicit limits; preserve baselines and generic estimates.

### E8 — Topologies, shared subcomputations and graphs

Local discovery checkpoint: `experiments/block_sparse_cuda/topology_reuse` now
inventories bounded pre-special prefixes using the actual `diag.py` pairing
generator and production block-path visitor. The four-mode `prefix-01` cohort
finds candidate operator/convolution overlap for number symmetry at orders 3/4/6,
but no overlap for the selected order-6 mixed paths or periodic zero masks.
This is structural evidence only: no numerical memoization or graph speedup is
qualified. Next independently replay shared prefixes, invalidate on changed G,
coefficients, mesh and masks, then time complete calls with bounded cache storage.
The first-connected/first-index selection is biased; broaden selections before
making a workload-wide decision. Current allocation/synchronization/cutoff paths
require refactoring before whole-call capture. GH200-class hardware remains the
target; the local RTX A4000 inventory does not qualify target performance.

The subsequent `replay-01` control independently advances CSR/hoisted-DLR and
dense-operator/original-CPPDLR prefixes. Across the 24 local cases, 384 repeated-key
comparisons are exactly equal and independent-reference maximum absolute error
is 8.424e-18. Retained candidate payload stays below 205 kB (explicit cap 64 MiB).
Fresh changed-G/coefficient replays change contents, confirming that structural
identity alone cannot justify numerical cache hits. This qualifies the selected
CPU prefix keys only; no device DAG, input-update cache protocol, complete-call
timing or CUDA graph is implemented. Next compare a bounded call-local device
cache against independent complete calls, preserving special/cutoff/reduction order
and charging discovery, cache misses and retained memory. Broader index selections
and target-hardware qualification still govern any production decision.

Private device checkpoint `cache-01` implements an 8 MiB/4096-entry post-edge cache
and exact full-G snapshot invalidation (32 MiB host snapshot cap, independent device
execution beyond it). Eighteen paired processes/162 samples preserve special,
cutoff and output order and match independent complete per-topology calls exactly.
On the local Ordered/Mapped route, order-3 number cohorts improve 21.9% with G
changed every sample (37.8% unchanged replay); order-6 mixed changed-input cohorts
regress 0.6%. Thus graph-free sharing can pay on selected cohorts, but is not a
universal default. Eight inherited evaluator controls and both-policy memcheck pass.
Next widen index/cohort choices, force cache saturation and tails, and measure the
selection/setup cost of enabling reuse. Keep the private cache separate until these
gates and target-hardware evidence justify a production policy. CUDA graphs remain
conditional and are not implemented by this cache experiment.

Cache-limit checkpoint: private configurable lower limits now force byte rejection,
one-entry saturation and snapshot bypass on all three four-mode partitions.
Exact per-call controls cover uneven first-four-index ranges, replay, changed G,
zero masks and restoration for both Library/Reference and Ordered/Mapped routes.
Oversized settings reject without altering state; explicit clear releases host
snapshot capacity as well as device entries. This closes forced-limit and selected
range-tail controls, not broader cohort selection, setup-policy or GH200 gates.

Broader private checkpoint `selection-01`: distributed connected selections within
15/105/4096 scanned pairings, first indices 0/7/31 (count three), three four-mode
partitions and one six-mode number case yield 28 processes/252 exact paired samples.
Both accepted and contributing path counts match, including output-cutoff rejection.
Changed-G active order-3 gains are 18–27% locally; the six-mode number case gains
21.5%. Order-6 mixed selections remain roughly neutral/slower, so keep a selective
policy rather than unconditional reuse. The bounded generator prefix is not a
population sample; topology discovery/model setup remain outside device timing.
Next measure the cost of a reuse selection policy against always-on/off execution
and wider/random index cohorts. These local results do not qualify GH200 performance
or settle the remaining full-function memory and many-small batching decisions.

Production E7 checkpoint: compact DLR liveness is promoted after the private controls
and full-call experiments. Completed solve RHS is released before convolution;
disjoint output holds diagonal scratch, while aliases retain a separate diagonal.
Retained square convolution scratch is four tiles and next-solve peak five; the
production evaluator reserve drops from six to five maximum-size tiles. Tables,
arithmetic order and public signatures are unchanged. Original cached DLR is retained
as `compact_dlr/historical_dlr.*`; private NCA admission retains its old six-tile
reserve. For the prior N=14/rank26 analytic inventory, variable generic storage drops
from 90.409 to 85.845 GiB; this is neither an allocation measurement nor full-system
qualification and still excludes fixed/operator/host/opaque storage. CPU 80/80,
main CUDA 101/101 and native-disabled 86/86 suites pass. Production alias/storage
memcheck reports zero errors/leaks. Two-rank integration passes six methods per rank.

Private E7 complete-output checkpoint: `experiments/block_sparse_cuda/output_sectors`
keeps one self-energy output sector on device and the complete output on host,
preserving path order and performing every accumulation on GPU. Full propagator
inputs remain resident. The device estimate replaces one full function with one
maximum tile; the prior N14/rank26 variable estimate falls another 10.979 GiB to
74.867 GiB, excluding fixed/operator/host/opaque storage. Full output host payload
has a configurable budget checked before allocations/uploads, including empty ranges.
Nine controls pass in main and native-disabled configurations, with exact full-sector
outputs, changed masks, recovery and a device budget rejected by resident evaluation
but accepted by streaming. Focused memcheck reports zero errors/leaks. This is not
promotion or actual N14 qualification: transfer/setup cost, measured peak storage,
input/slab streaming and target-hardware capacity remain open.

E7 measured complete-call checkpoint `output_sectors/outputs-02` adds an explicit
return-ownership control to the resident and streamed routes. Seven physical cases
through ten modes yield 147 exact output/count rows with rotated route order. Ten-
mode changed-G resident/explicit-move/streamed medians are 204.181/162.227/171.698 ms;
streaming costs 5.8% against the ownership control, despite saving 44,620,736 bytes
in its explicit bound. Sampled device deltas also fall, but they are global selected-
point diagnostics, not continuous/private peaks. Prefer resident execution when it
fits; investigate the return move as a separate production improvement, preserving
the copy-return baseline. Streaming remains a capacity route pending independent
peak/host measurements, transfer/placement policy and GH200-class qualification.

Production ownership checkpoint: self-energy returns its full host vector by explicit
move. CopyOutputEvaluator remains generated as a copying control; streaming/copy/move
experiments retain their meaning. Main four focused CTests and native-disabled three
pass, covering production observables, Python integration and complete-output controls.
CPU-only library compilation and focused memcheck pass. `outputs-03` has 147 exact
paired rows; local changed-G eight/ten-mode production medians improve 11.94%/21.53%
against copying. Device arithmetic, bounds and public API remain unchanged. Full
production suites/MPI remain the preceding compact checkpoint; no new full-suite or
target-hardware claim is inferred from this ownership qualification.

Production C++ placement checkpoint: explicit `OutputPlacement::HostSectors` selects
the qualified single-output-sector route; Resident stays default. Host output payload
budget is checked before upload even for empty ranges, and the SE device estimate
substitutes one maximum tile for the full output function. All required sectors are
returned; GPU arithmetic and per-sector accumulation order are unchanged. Inputs
remain resident and other observables retain their bounds. Eight reusable controls
run with HostSectors forced; production admission tests cover tight accepted/rejected
budgets, masks, empty ranges and invalid enums. The experiments use a pre-placement
source/header snapshot to preserve their baselines. Next expose placement/budget in
the consolidated Python facade and Solver/MPI configuration, then complete broader
capacity/host/target qualification. This is explicit placement, not automatic fallback
after CUDA submission or a solution to input/contraction tiling.

Python/MPI placement checkpoint: consolidated static/dynamic constructors expose
`output_placement='resident'|'host_sectors'` and `host_output_budget`. Solver stores
`block_sparse_output_placement`/`block_sparse_host_output_budget`, forwards them to
CUDA, validates them without importing CUDA for CPU configurations, and retains them
in deepcopy/HDF archives. Old archives default to Resident/512 MiB. Seven Python
methods pass on both CUDA configurations; CPU metadata/archive control passes with
six GPU skips. Two ranks pass seven methods each, including rank-local invalid
placement, zero host budget, collective rejection and reusable later evaluations.
Existing module/error orchestration remains consolidated. Target/host peak and
larger capacity, many-small batching and cutoff qualification remain open.

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

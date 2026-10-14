# Block-sparse CUDA work log

## 2026-10-14 — Replay-heavy cohorts and owned input reuse

Previous turn was progress: adjacent high-order cutoff mismatch diagnosed; separate
replay experiment qualified. Reviewed replay/admission. Replaced whole-G copy per
sensitive lane with one lazy owned copy per cohort, retaining it until cohort fan-in
finishes; no cross-call cache. Maximum live payload/admission reserve is unchanged.
Added order-two cutoff stress: isolated-path calibration, six-mode/eight-index range,
seven surrounding amplitudes, zero and two nonzero poles. More than512 structural
paths plus a tail, contributing-count/output checks and exact repeated output.

Initial two-test run passed37.576 s. Added a private experimental replay observer
and reset, proving actual >512 replayed lanes at the zero-pole center rather than
just structurally accepted paths. Repeated calls must have identical replay counts.
Full replay CTest passed46.29 s, detailed LastTest proves17 active GoogleTests.
Ran memcheck once on the replay-heavy cohort test: passed482.049 s, zero errors and
zero leaked bytes. Long instrumentation was repeatedly verified live through the
tool session/PIDs and CPU/GPU activity; no restart or overlap. Only after terminal
completion rebuilt/reran preserved constants suite, which passed. All handles
terminal before checkpoint. Logs /tmp/xca-bs-scalar-cutoff-{cohorts-build,cohorts-test,
count-build,count-suite,cohorts-memcheck,cohorts-baseline-build,cohorts-baseline-test,
cohorts-lasttest}.log. Compact evidence/hashes retained in local cohorts-01 folder.

Code and this plan/log/evidence checkpoint are separate commits. Goal active:
bounded replay ownership/cohort/pole semantics strengthened, but fixed uncertainty
band remains unproven for arbitrary rank/conditioning/cancellation/order. Next
resolve conservative promotion policy and broader scalar operation capability;
target new-variant and Python/MPI/capacity gates remain. No performance inference
from instrumentation. No new Arrhenius jobs (.08361 nominal GPU-hours cumulative),
install, push or unrelated-work changes.

## 2026-10-14 — Adjacent high-order cutoff replay experiment

Previous turn was progress: active order-six/ownership oracle controls passed. Added
a stricter reference-defined threshold search to adjacent representable amplitudes,
then probed 17 neighbors around each endpoint. Original constants candidate failed:
around amplitude .10775810972137165 it discarded a ~1e-16 contribution retained by
general CUDA. This is a contributing-path semantic mismatch, not ordinary output
roundoff that can be accepted by a larger tolerance. Retained the failure evidence.

Added separate scalar_cutoff folder/CMake evaluator, preserving earlier candidate
and production. Experimental band abs(magnitude-1e-16)<=1e-26 triggers general CUDA
lane recomputation using owned copies of input views and the general vertex/edge/
special/prefactor sequence. Replaces lane data before normal cutoff and ordered
fan-in; no CPU evaluation, atomics or reordered accumulation. Extra owned input
function and returned scalar lane enter admission. Enabled adjacent test only for
this variant; earlier baseline remains available with known boundary limitation.

Adjacent replay control passed 1.574 s. Final constants/replay CTests passed 8.81/
9.97 s, total18.79 s (15/16 active cases). Focused replay cutoff/admission memcheck
passed two tests in12.971 s with zero errors/leaks. After all handles terminated,
five-sample ordinary N=6 scalar range timings passed CPU screen/exact replay,
matching 640 accepted paths and error6.381e-17. Median constants/replay7.484/7.327 ms:
no evident regression in this single screen, no speedup or borderline-cohort cost
claim. Evidence/hashes retained in scalar_cutoff/thinkstationP620_RTXA4000/replay-01.
Code fac46e1 and evidence4931846 precede this separate plan/log checkpoint.

Goal active. Fixed band is a diagnostic policy, not a proof for arbitrary rank,
conditioning, cancellation, shapes or orders. Next stress numerical coverage and
replay-heavy capacity/cost before promotion; broader scalar sets and Python/MPI/
target/capacity gates remain. No new Arrhenius jobs; cumulative target allocation
301 s (.08361 GPU-hours nominal). No install/push or unrelated-work changes.

## 2026-10-14 — Active high-order scalar arithmetic and ownership

Previous turn was progress: GH200 scalar candidate qualified 14 semantic controls
and matched timing cohort, but order-six window rejected every path. Kept further
work local to preserve allocation. Added active order-six chain qualification with
bounded metadata discovery (two modes, one zero pole, selected index 10). Asserts
structural and arithmetic activity, compares contributing counts, brackets the
completed-path cutoff with 24 probes and checks both sides at explicit 1e-4 amplitude
margins. Separately allocated changed propagators keep old owners alive; reusing
the old owner verifies the evaluator reads current inputs rather than cached pointers.

Initial inherited 1e-13 output comparison failed: high-order differences are tiny
absolutely but up to several e-13 relative. Counts agreed. Investigated with an
independent long-double scalar self-energy oracle built from existing qualified
precision primitives, retaining cached double LU/rank/hybridization constants and
shared connectivity. Oracle measures reference 4.28209e-13 and candidate 3.89703e-13
relative error. Both routes accumulate rounding; no arithmetic bug or tolerance
change to older tests/production is inferred. New order-six control checks each route
at 1e-12 against the oracle, matching the previously documented high-order diagnostic
approach. It does not independently qualify traversal or DLR approximation accuracy.

Focused constants test passed .796 s. Built all four scalar variants sequentially
in the same tree, then their CTests passed 8.78/8.69/8.75/8.71 s, total 34.94 s.
Constants memcheck of active order-six/changed-address control passed 4.322 s with
zero errors and zero leaked bytes. All process handles terminal before checkpoint.
Logs /tmp/xca-bs-scalar-highorder-{build,test,oracle-build,oracle-test,variants-build,
variants-test,memcheck}.log. Test/helper documentation committed separately from
this plan/log checkpoint. No new Arrhenius jobs; target allocation remains 301 s,
nominal .08361 GPU-hours. No install/push or unrelated-work changes.

Goal active. This closes a bounded active high-order case, not ULP-adjacent cutoff
equivalence for arbitrary paths. Next investigate that stricter boundary behavior
and broader poles/topologies/irregular scalar sets before production promotion;
target rerun of new controls and Python/MPI/capacity gates remain outstanding.

## 2026-10-14 — GH200 scalar batching qualification

Previous turn was progress: committed initial target matched baseline. Inspected the
retained scalar experiment, its constants/arena/fan-in variant and exact matched CPU
control. Added scalar cohort selection to the archive-aware recorder, preserving
initial defaults. Prepared a one-GPU/fifteen-minute job reusing pinned 3f462c3 Arm
tree. Selected six cases across unchanged CPU/ordered GPU/batched CPU/batched GPU:
zero/active order-three windows, N=6 eight-index range, N=10 scalar order two,
order-six chain window and mixed retarded correlator fallback. Syntax/help/diff and
remote scheduling checks passed; exactly one job submitted, 3819263.

Job COMPLETED exit 0 in 90 seconds, one GPU, billing TRES 1, nominal .025 GPU-hours.
Built candidate and matched CPU, then all 14 active GoogleTests passed (7.38 s CTest).
Controls cover evaluator observables, multi-cohort tails, changed masks, NCA cutoff
adjacency, admission/empty ranges, multi-field general route, ordered cancellation
and coefficient layout/updates. All 24 benchmark processes/72 samples passed finite,
CPU-screening and exact repeat checks. Matched CPU equals original exactly. Structural
counts/ranges/rank match ordered versus candidate; largest absolute error 1.955e-16.

Range ordered GPU 732.530 ms becomes 6.739 ms (108.7x), matched CPU 6.071 ms.
N=10 ordered 3362.550 ms becomes 17.587 ms (191.2x), matched CPU 18.371 ms (only
1.045x). Active smaller window candidate 3.897 ms versus matched CPU 3.116 ms;
original CPU .806 ms. Mixed correlator falls back as intended, ~10.8 ms for both GPU
routes. Order-six and zero windows reject all paths, so high-order active arithmetic
is still unqualified. Three samples/one allocation cannot establish robust crossover
or whole Solver throughput. No production promotion, binding or default change.

Committed recipe, benchmark evidence and plan/log separately. Raw samples and hashes,
semantic LastTest.log and accounting retained under Arrhenius_GH200/scalar-01 in
CPU/CUDA/common benchmark folders; full remote logs/cache retained. Batch RSS reported
36 MiB is not interpreted as whole child-process peak. Total target GPU allocation
301 seconds across three jobs, nominal .08361 GPU-hours. Goal active: next close
active high-order/cutoff and broader scalar capability/lifetime gates locally before
additional target runs, then continue Python/MPI and capacity qualification. No
install/push; baselines/experiments and unrelated user work preserved.

## 2026-10-14 — Initial matched GH200 baseline

Previous turn was progress: first target C++ smoke qualified 20 GoogleTests and
committed recipe/evidence separately. Inspected complete-call fixtures/frontends,
selected seven cases (scalar zero/active windows, mixed retarded correlator, number
retarded SPGF, order-three SE, N=8/N=10 NCA). Added a bounded recorder with source
archive revisions, runner/binary hashes, raw samples, 120-second process timeouts and
immediate failure stop. Reused the pinned 3f462c3 Arm source/build, enabling benchmarks
and building only unchanged CPU/library CUDA/ordered CUDA frontends. No source-tree,
shared installation or dependency mutation. Shell syntax/help/diff checks passed;
sbatch --test-only succeeded before exactly one submission, job 3818907.

Job completed exit 0 in 91 seconds, one GPU, billing TRES 1, nominal 0.02528 GPU-hours.
No additional CPU/memory request; 72 CPUs provided, builds capped at six. Batch RSS
1,062,976 KiB. All 21 processes and 63 timed samples passed numerical/repeatability
screens; maximum absolute error 2.865e-14. Retrieved raw samples/manifests/accounting
to CPU/CUDA/common benchmark folders, machine Arrhenius_GH200, campaign initial-01.
Recipe and benchmark evidence commits are separate from this plan/log checkpoint.

Ordered CUDA median N=8/N=10 NCA 67.557/169.533 ms versus CPU 98.369/4827.384 ms:
1.46x/28.47x. Scalar active 8.511 ms versus CPU .815 ms; mixed retarded correlator
11.039 versus 1.059 ms. N=6 retarded SPGF 67.671 versus 22.754 ms; N=6 order-three
SE 26.392 versus 27.965 ms, weak signal. Scalar first=0 has zero signal and rejects
all 64 paths; first=1 accepts four. This reinforces the many-small batching gate;
it does not establish dispatch thresholds, algorithm-matched CPU superiority, whole
topology-sum/Solver speedup or full N=14 qualification. Three samples/allocation and
single indexed crossing topologies are explicit limitations. No backend policy changed.

Goal active. Next qualify retained scalar-path batching on GH200 with active/rejected
windows and exact semantic controls, and continue Python/MPI/capacity gates. Total
target allocation so far 211 seconds, nominal .05861 GPU-hours, within user's shared
100-hour monthly ceiling. No install/push. Full remote logs/cache retained.

## 2026-10-14 — First GH200 smoke qualification

Previous goal turn was progress: committed the user's 100 GPU-hours/month constraint.
Used Arrhenius/Arrhenius-software workflows, isolated source archive 3f462c3 and pinned
GTest 988ea2c. Preserved older shared checkout, CPU/CUDA installs and environments.
Cleaned inherited project CPU paths and CUDA driver stubs before Arm modules/TRIQS.
First transfer failed because the parent qualification directory was absent; created
the selected directory and retried transfer only. No compute was spent on that error.

Shell syntax and remote sbatch --test-only passed. Scheduler grants 72 CPUs per GPU;
build worker count uses allocated CPUs capped at six, without extra resource requests.
Submitted exactly one job, 3818566, one GPU/ten-minute cap. projinfo reported 2.79 GPU
hours over the preceding 30 days and warned its estimate may be unreliable. Polling
confirmed RUNNING, then COMPLETED exit 0. Actual allocation 120 seconds, nominal
0.0333 GPU-hours, billing TRES 1; batch MaxRSS 828 MiB. No duplicate/retry compute.

GCC 14.3.0/CUDA 12.9.86/aarch64 configure and focused build passed with native BS ON,
dense-small native OFF, architecture 90, Python OFF. Four CTests passed in 10.63 s:
admission/device/evaluator/DLR. Retrieved LastTest.log proves 20 active GoogleTests
(3/5/8/4), not disabled CUDA cases. Numerical tests cover focused SE/SPGF/correlators.
Empty CMake Git hash is expected for source archives; revision JSON pins the source.
Warnings were the unused binding-update option and existing shadowing warnings.

Compact hardware-specific evidence committed separately from the job recipe;
full logs and reusable Arm build remain in shared/qualification/xca-block-sparse-3f462c3.
Goal active: no target speedup, full suite, Python/MPI or full N=14 qualification claim.
Next use the retained Arm build for a small matched benchmark cohort and broader
semantic gates within the declared one-GPU-hour initial cohort cap. No install/push.

## 2026-10-14 — Arrhenius allocation constraint

User specifies only 100 GPU-hours/month on Arrhenius. Updated the plan to treat this
as a shared allocation ceiling, screen experiments locally, reuse Arm builds, and
start with bounded single-GPU smoke and representative cohorts. Initial caps are
ten minutes for build/smoke and one GPU-hour for the subsequent benchmark cohort,
not authorization to consume the full monthly allocation. Record actual allocation,
elapsed time and charges, including failed jobs, and inspect remaining balance.

Arrhenius read-only checks confirmed SSH access, the GPU account and installed Arm
CUDA TRIQS. The shared xca checkout is older; preserve it and use an isolated source
snapshot for qualification. A smoke script and source-only staging were drafted;
no job has been submitted and no GH200 numerical or performance qualification is
claimed. Existing implementation goal remains active. This checkpoint changes only
the plan and work log; qualification drafts remain separate and uncommitted.

## 2026-10-13 — Investigation and first draft

Persistent goal: develop an evidence-based CUDA experiment plan covering many small
blocks, mixed blocks, and 10/14-mode total-number sectors; adversarially review and
revise it. Work is on `dev_cuda`. This is planning work; no block-sparse CUDA
implementation or performance experiment has been completed.

Reviewed evaluator traversal, special vertices, correlator allocation, atom_diag
basis conversion, symmetry sets, dynamic-operator restrictions, flat indexing,
topology enumeration, MPI orchestration and the installed cppdlr convolution.
Reviewed dense-native ownership/benchmark lessons and retained all experiments.
Checked installed CUDA headers and NVIDIA cuBLAS, cuSPARSE and graph documentation.
The plan links the source areas and documentation used.

Main findings:

- Fundamental operators are signed partial permutations in the evaluator's Fock
  basis. Direct atom_diag state-map construction avoids dense eigenbasis round trips.
  Barred and retarded operators need weighted/general sparse representations too.
- Large Q-only propagator blocks remain dense. At N=14,r=26 one all-sector function
  needs 15.542 GiB before output/workspace. Current padded evaluator/correlator buffers
  cannot simply be copied onto the GPU. Raw dense quartet payload at p=2 is 46.866 GiB.
- Current complex-double library capability does not support generic heterogeneous
  grouped GEMM; deterministic sparse algorithms have batching/layout limits.
- Topology enumeration, flat-index overflow, dead-path rates and replicated model
  memory are separate limits; a fast primitive does not qualify a full call.
- The installed CPU convolution repeats rank transforms within the DLR-node loop.
  An algorithm-matched CPU control is needed to attribute speedups honestly.

Successful bounded probes: real atom_diag metadata for N=4/6/8/10 in occupation and
Q-only partitions; direct-Fock versus rotated operator comparison at N=6. No N=14
atomic diagonalization, full high-order sum, CUDA prototype, or cluster job was run.
The plan separates measured results from analytical estimates.

The scripts below preserve the exact recipes rather than depending on temporary
files. Run separately with `source ~/env/xca-cpu.sh`, `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`; each completed under a 45-second limit. The first reports
single-run setup/metadata times, not benchmark statistics. TRIQS installation:
`~/apps/triqs`; local source baseline `d1fa7f7`.

### Atom_diag inventory recipe

```python
import json,math,time
from collections import Counter
from triqs.atom_diag import AtomDiag
from triqs.operators import n,c,c_dag
out=[]
for N in (4,6,8,10):
 for regime in ('occupation','number'):
  start=time.perf_counter()
  fops=[('f',i) for i in range(N)]
  Q=sum(n('f',i) for i in range(N))
  H=sum((.13*(i+1)+.007*i*i)*n('f',i) for i in range(N))
  H+=sum((.17+.003*(i+j))*n('f',i)*n('f',j) for i in range(N) for j in range(i))
  if regime=='number':
   for i in range(N):
    for j in range(i):
     z=.011*(1+i+j)+.009j*(i-j)
     H+=z*c_dag('f',i)*c('f',j)+z.conjugate()*c_dag('f',j)*c('f',i)
  ad=AtomDiag(H,fops) if regime=='occupation' else AtomDiag(H,fops,[Q])
  dims=list(map(int,ad.get_subspace_dims()))
  connections=[tuple(ad.c_connection(i,b) for b in range(ad.n_subspaces)) for i in range(N)]
  groups=Counter(connections)
  out.append(dict(N=N,regime=regime,block_histogram=dict(Counter(dims)),blocks=len(dims),max_dim=max(dims),symmetry_set_sizes=sorted(groups.values()),seconds=time.perf_counter()-start))
print(json.dumps(out,indent=2))
```

### Direct fundamental-operator comparison recipe

```python
import numpy as np,json,math
from triqs.atom_diag import AtomDiag
from triqs.operators import n,c,c_dag
N=6;fops=[('f',i) for i in range(N)];Q=sum(n('f',i) for i in range(N))
H=sum((.13*(i+1)+.007*i*i)*n('f',i) for i in range(N))
H+=sum((.17+.003*(i+j))*n('f',i)*n('f',j) for i in range(N) for j in range(i))
for i in range(N):
 for j in range(i):
  z=.011*(1+i+j)+.009j*(i-j)
  H+=z*c_dag('f',i)*c('f',j)+z.conjugate()*c_dag('f',j)*c('f',i)
ad=AtomDiag(H,fops,[Q]);maxerr=0.;stored=0;structural=0;outside=0.
for i in range(N):
 for b in range(ad.n_subspaces):
  target=ad.c_connection(i,b)
  if target<0:continue
  states=list(map(int,ad.fock_states[b]));target_states=list(map(int,ad.fock_states[target]));positions={s:j for j,s in enumerate(target_states)}
  exact=np.zeros((len(target_states),len(states)),complex)
  for col,state in enumerate(states):
   if state & (1<<i):exact[positions[state^(1<<i)],col]=(-1)**((state&((1<<i)-1)).bit_count())
  U=ad.get_unitary_matrix(b);V=ad.get_unitary_matrix(target)
  reconstructed=V@ad.c_matrix(i,b)@U.conj().T
  maxerr=max(maxerr,float(np.max(np.abs(exact-reconstructed))))
  stored+=exact.size;structural+=int(np.count_nonzero(exact))
  if np.any(exact==0):outside=max(outside,float(np.max(np.abs(reconstructed[exact==0]))))
print(json.dumps(dict(N=N,annihilation_stored_entries=stored,structural_nonzeros=structural,expected_nonzeros=N*2**(N-1),max_fock_reconstruction_error=maxerr,max_structurally_zero_entry=outside),indent=2))
```

Inventory output: occupation blocks 16/64/256/1024 of size one; Q-only maxima
6/20/70/252 with one orbital symmetry set of size N. Q-only elapsed setup/metadata
was approximately 0.0005/0.0043/0.0939/2.2578 seconds for N=4/6/8/10.
Operator output: 4752 dense entries, 192 structural nonzeros,
maximum reconstruction error 2.109491387119662e-15 and maximum spurious zero-entry
magnitude 1.0612982058267527e-15. This is a structural feasibility check, not
high-order numerical qualification.

The first draft is followed by a separate adversarial self-review, not an independent
model review. Plan/log changes are committed separately from code. No production
code changes are needed for this task. Unrelated worktree changes are preserved.

## 2026-10-13 — Adversarial review, revision and completion

Initial draft commit: `58dde4b` (`[docs] plan block-sparse CUDA experiments`).
A separate adversarial self-review challenged the draft against the source and
probe evidence. Section 8 records the challenges and dispositions. Revisions add
physical rank/pole sensitivity, full-output memory accounting, the extra correlator
buffer, permuted mode/state-order tests, low/zero-survival sampling, and a distinction
between shared intermediates and reassociating the topology sum. Corrected a source
link found by validation. Unresolved performance choices remain explicit experiments.

Validation completed:

- All local Markdown links resolve; both embedded probe recipes parse.
- Exact binomial identities, all six memory-table rows, high-order index overflow
  example and topology pairing count were recomputed successfully.
- Both documented probes reran successfully with `python3`, CPU environment and
  one numerical-library thread. All inventory structural fields matched, and the
  six-mode operator errors/counts reproduced exactly. Temporary repeat results are
  `/tmp/xca-block-review-inventory.json`; durable recipes/results are above.
- `git diff --check` passed. Only the plan and this log are included in task commits;
  existing `.gitignore`, AGENTS files and untracked tests are preserved.

No build or full CTest run was warranted by documentation-only changes. No CUDA
performance claims or complete N=14 qualifications are made. The planning goal is
complete after the reviewed documents are committed; implementation experiments
remain future work. No new quota supervisor configuration was performed by this task.

## 2026-10-13 — Symmetry-set limitation

Added the user's observation that the current symmetry-set analysis can be overly
restrictive, including multiorbital density operators with scalar Hilbert-space
blocks. The plan now calls for revisiting grouping restrictions without documenting
the constant-offset workaround. Documentation only; `git diff --check` passed.

## 2026-10-13 — Persistent implementation goal activated

The user authorized a new persistent goal to implement `plan_block_sparse_cuda.md`.
The earlier planning goal is complete; this implementation goal is now active.

Deliver an operational block-sparse CUDA evaluator covering the existing ordinary
SE, SPGF, one-time correlator and dynamic-interaction functionality. Follow the
experiment gates and update the design from measured evidence rather than choosing
kernels or graph policies in advance. Include the current symmetry-set limitations
in the investigation. GH200-class accelerators remain the performance target;
local RTX A4000 results are development evidence with explicit qualification limits.

Start with E0 bounded workload/memory inventory and E1 direct sparse operator export
from atom_diag Fock metadata, including a CPU numerical control. Then establish E2's
allocation-bounded device reference before optimizing or expanding system size.
Use the existing consolidated CUDA module, range APIs and Python MPI orchestration.
Preserve dense baselines, all experiments and unrelated user work. Commit verified
code continuously on `dev_cuda` with terse messages; commit plan/log checkpoints
separately. Do not push. Do not mark the implementation goal complete until its
supported functionality and required verification are achieved.

This checkpoint records goal activation; it does not claim that new implementation
experiments or quota-supervisor reconfiguration have already been performed.

## 2026-10-13 — E1 initial exact sparse-operator foundation

Implementation commit: `fc54948` (`[bs] export sparse Fock operators`).
The persistent implementation goal remains active; this is an initial E1 milestone,
not a completed CUDA evaluator or complete experiment campaign.

Added portable host model code in `c++/triqs_xca/block_sparse/cuda/sparse_operator.*`:
owned CSR blocks, exact many-body operator export using atom_diag Fock metadata,
adjoint construction, structural validation and CPU left/right sparse applications.
Export holds sparse entries rather than dense operator matrices. Explicit dense
expansion is used only for small numerical controls. Existing evaluators and
experiments are unchanged.

Reuse TRIQS's imperative fermion-action machinery with a small exact-state adapter.
The standard TRIQS sparse-state iterator prunes through `utility::is_zero`; the new
adapter omits only exactly zero amplitudes. Unit monomials are compiled separately
and original coefficients are applied afterward, retaining tiny input coefficients.
Connections are determined before coefficient cancellation, matching the existing
monomial-based policy. Interaction validation preserves zero-operator and injective-
connection restrictions without allocating a throwaway adjoint.

Source investigation corrected an important assumption: atom_diag's
`get_matrix_element_of_monomial` applies the whole monomial in full Fock space and
projects the final state. Intermediate c/cdag connections do not implement that
policy for occupation-cutoff instances. The exporter and a regression test now cover
this case; the plan records it. The exporter explicitly checks the underlying
TRIQS full-Hilbert-space int-index limit before construction; 10/14 modes fit that
index representation, without implying their complete atomic setup is inexpensive.

Six GoogleTests cover complex mixed blocks, real scalar blocks, noncanonical mode
labels/insertion order, exact signed maps, independent bit-sign and anticommutation
identities, adjoints, arbitrary rectangular left/right products, tiny coefficients,
exact cancellation with surviving connections, final-state projection, unsupported
multi-target/non-injective operators, empty operators and invalid CSR metadata.
A six-mode Q-only model has 192 exported annihilation nonzeros, as predicted.

The complete CPU numerical control packs exact exported bare/interaction operators
into a bounded legacy quartet and uses the unchanged CPU traversal. It compares SE,
SPGF and one-time correlators with the existing rotated-operator CPU evaluator at
orders 1/2/3, with dynamic density and non-Hermitian hopping channels, up to 64 flat
indices per topology and a relative 1e-13 bound. Selected SE outputs must be nonzero.
This small control does materialize dense blocks; the production exporter does not.
Barred operators in this control are rebuilt from exact operands by the existing
quartet implementation. Production sparse barred materialization remains pending.

Verification: new sparse test executable and existing block-sparse dynamic-evaluator
and evaluator-range regressions passed, 3/3 CTest entries in each CPU and CUDA build
(3.09 s CPU; 4.34 s CUDA). New sparse tests are host tests in both environments;
they do not exercise new CUDA arithmetic. `git diff --check` passed. Logs are under
`/tmp/xca-bs-sparse-{cpu,cuda}-{configure,build,test}.log` and may disappear.

Next: E0 reproducible workload/path traces and checked memory admission; complete
E1 model packing/barred-operator and broader numerical qualification; then E2's
allocation-bounded device reference. No N=14 diagonalization, GH200 qualification,
new device benchmark, Python CUDA backend registration or supervisor reconfiguration
has been performed. Preserve the full implementation goal and its remaining gates.

## 2026-10-14 — E1 model packing and E0 bounded path inventories

Code commit: `e08f730` (`[bs] pack sparse models and trace paths`). Benchmark
commits: `de527f1` (`[bench] inventory block-sparse workloads`) and `922d9c1`
(`[bench] record block-sparse inventories`). The implementation goal remains active.

OperatorModel now owns exact fields/adjoints and both pole-weighted barred variants,
coefficient snapshots and current symmetry grouping. It validates coefficient shapes,
finite values and cross-set compatibility before materialization. Checked storage
bounds precede barred allocation; host-owned capacity and CSR payload are separately
reported. These exclude atom_diag, allocator overhead and construction temporaries.
The bound is not a complete host/device admission policy.

The streamed self-energy path collector preserves CPU index/output/set order,
operator connections and input-dependent propagator masks. It checks proposal-count
arithmetic before calling a visitor. Visitors borrow a reusable descriptor/current
Backbone; retention requires a copy. A bounded metadata-only observation hook in the
CPU evaluator supplies an independent traversal control without executing arithmetic.
New tests compare complete path descriptors for ordinary/retarded models, both masks,
orders 1–3, range boundaries and recovery after trace-limit failures. Operator-model
tests compare complex barred matrices with the existing quartet and check independent
analytic memory/overflow controls. Existing numerical arithmetic remains unchanged.

Full verification after core changes: CPU 71/71 CTest entries passed in 50.19 s;
CUDA 79/79 passed in 104.94 s. Logs:
`/tmp/xca-bs-model-path-{cpu,cuda}-{configure,build,test}.log`, where applicable.
The new host-only tests do not qualify new GPU arithmetic. Rebuilding all bindings
was necessary because the CPU observation hook added private evaluator fields.

E0's first reproducible inventory lives in `benchmarks/block_sparse_common` and
`benchmarks/block_sparse_cpu`. It uses deterministic occupation, total-number and
mixed partitions, ordinary plus retarded channels, crossing/chain topologies and
three bounded 32-index windows. The recorder stores commands, hashes, machine
metadata, return codes, raw summaries and shape histograms. All 144 actual N=4/6/8/10
cases and six analytic N=10/14 cases succeeded in 56.65 s summed subprocess time.
Results are under `benchmarks/block_sparse_cpu/thinkstationP620_RTXA4000`; the hardware
README explains scope and timing units. Shape inventories distinguish per-rank matrix
products, separate rank-solve RHS widths and the non-GEMM special mixing reduction.

For N=10, the first ordinary crossing window yields 29,696 accepted scalar paths out
of 327,680 proposals (4.660 ms null traversal), versus 256/352 total-number paths
(0.015 ms) and 1,246/2,304 mixed paths (0.065 ms). Largest blocks are respectively
1, 252 and 100. This identifies distinct scheduling regimes, not kernel winners.
Analytic N=14 total-number/rank-26 capacities reproduce the independent test oracle:
one function 16,688,505,600 bytes; largest rank buffer 4,899,907,584; legacy scratch
88,198,336,512; initial correlator sides 146,997,227,520. These are incomplete peak
estimates. No N=14 diagonalization or device allocation occurred.

Found a second CPU numerical cutoff: each completed self-energy path is accumulated
only if max(abs(T_out)) > 1e-16. The plan now requires preserving/testing it separately
from whole-propagator-block pruning. Structural accepted paths include numerical
zeros and cutoff-discarded contributions; batched code must not apply the cutoff to
an aggregate. Any policy change needs separate numerical qualification.

Remaining: numerical phase timing, weighted work/bytes and measured live-memory
admission (E0); broader cancellation/update-cost controls (E1); allocation-bounded
owned-stream device arithmetic for rectangular sparse products and DLR transforms,
then all observables (E2). GH200 validation, backend integration and later optimization
gates remain pending. No supervisor reconfiguration, install or push was performed.

## 2026-10-14 — E2 initial owned-stream rectangular device primitives

Code commit: `9be8aa4` (`[bs] add CUDA rectangular products`). The persistent
implementation goal remains active. This is a device primitive milestone, not a
completed block-sparse evaluator or E2 qualification.

Added `block_sparse/cuda/device_ops.*`, guarded by NDA_HAVE_CUDA independently of
native CUDA-small support. DeviceOps owns its nonblocking stream and cuBLAS/cuSPARSE
handles, checks the construction device and uses explicit default-stream upload
events. Public calls complete synchronously. Inputs uploaded on other independent
streams must be made ready by their owner. It supports rectangular per-node products,
rank-axis GEMMs, zero/copy/axpy and rejects incompatible layouts/shapes/overlaps.

DeviceSparseBlock snapshots immutable CSR plus a plain-transposed CSR for right
products. It uses complex-double deterministic CSR_ALG3 one DLR node at a time,
without dense expansion or confusing transpose with adjoint. Sparse zero operators
use the explicit zero path. Descriptors survive until their node completes. Queried
SpMM workspace is checked before allocation, retained for reuse and released before
replacement so growth does not temporarily retain two workspaces. The per-block
storage budget counts both CSR payloads and retained workspace only; library handles,
opaque storage, host metadata, input/output and evaluator-wide buffers remain outside
it. Complete device admission is still pending.

Context lifetime follows ownership: snapshots retain the execution owner, while
cleanup selects the construction device before freeing descriptors, vectors and
handles, then restores the caller's selection. Operations on another device reject;
there is no automatic CPU replay after submitted device work or errors.

Four GPU tests cover complex non-square left/right products across rank/anchor
sizes, plain transpose semantics, snapshot independence, repeatability, reused
workspace, scalar and zero operators, budget/null-owner/shape/alias rejection,
recovery, rectangular dense/rank products and lifetime after releasing the caller's
owner pointer. Primitive absolute error is bounded by 3e-14 against independent CPU
products. The wrong-device cleanup branch is conditional on multiple devices and
was not exercised on this one-GPU workstation.

Verification: focused CTest passed in the regular CUDA build (0.58 s) and CUDA
library-only build with XCA_CUDA_SMALL=OFF (0.57 s). CPU-only build passed its guarded
entry, which runs a disabled test and does not qualify GPU behavior. Compute
Sanitizer memcheck/leak-check ran all four GPU tests: 0 errors and 0 bytes leaked.
`git diff --check` passed. Logs: `/tmp/xca-bs-device-{cpu,cuda,library}-*.log` and
`/tmp/xca-bs-device-memcheck.log`. No full-suite requalification is claimed for this
primitive-only addition; the preceding core checkpoint retains the 71/79 results.

Next: reusable owned cuSOLVER rank-r factors and rectangular RHS packing, time-ordered
DLR convolution/reflection and kernel weights, then the streamed SE and correlator
traversals with their numerical cutoff policies. No full-observable CUDA timing,
Python block-sparse CUDA registration, N=14 execution or GH200 qualification yet.

## 2026-10-14 — E2 rectangular DLR operations

Code commit: `be1535e` (`[bs] add CUDA DLR operations`). The implementation goal
remains active; complete CUDA observables, integration and optimization gates are
still pending.

Added `block_sparse/cuda/dlr_ops.*`: immutable cppdlr table/LU/pivot snapshots and
an owned cuSOLVER handle on the shared DeviceOps stream. The cached C-layout factors
are solved with plain transpose, matching NDA's factor convention, without forming
an inverse or refactorizing per edge. Status is explicitly copied from device storage
and checked after stream completion; NDA's global solver handles and synchronization
policy are not used. DeviceOps now supplies rectangular physical RHS pack/unpack,
scaling and complex rank weights (including exact in-place weighting).

Values-to-coefficients supports changing spatial RHS widths and in-place calls;
coefficient-to-values and reflection use the cached rank matrices. Time-ordered
convolution supports arbitrary compatible (M,K)*(K,N), including output aliasing
when shapes allow it. Separate products/additions retain cppdlr's reduction and beta
scaling order. Scratch consists of actual (r,M,K), (r,K,N) and three (r,M,N) buffers,
not padded squares. Exact-shape scratch/RHS storage is reused until shape changes.
Old buffers are released before replacements. Checked explicit storage admission
occurs before resizing, and growth rejection preserves existing workspace and caller
output. The reference rejects vector-count overflow before work; its current single
BLAS axpy/scal limit is INT_MAX elements. Chunked vectors may relax that limit later.

The budget covers five rank-square tables, pivots/status, RHS and convolution scratch;
it excludes caller inputs/output, host metadata and opaque library storage. Therefore
it remains a component budget, not complete evaluator/device admission. Scratch is
synchronous and intended for one owner at a time. Wrong-device cleanup follows the
previous ownership policy; multi-device execution remains unqualified locally.

Three DLR GoogleTests compare CPU cppdlr and CUDA within 1e-12 absolute error:
a small cached LU across scalar/rectangular/changing RHS widths, in-place solves,
reflection, real symmetric/nonsymmetric meshes at Lambda 10/40/120, several beta
values, rectangular convolutions and complete solve/convolve edge pipelines.
Cancellation-sensitive inputs, aliased output, repeated reuse, exact explicit-memory
counts, table/workspace-budget failures, unchanged output on rejected growth,
recovery and operation with NDA global solver synchronization disabled are covered.
The primitive test suite also checks separate/in-place complex rank weights.
Full diagram tolerance qualification still requires the next traversal milestone.

Verification: both new block-sparse device CTest entries passed in the normal CUDA
build (1.08 s) and CUDA library-only build with XCA_CUDA_SMALL=OFF (1.12 s). CPU-only
build passed both disabled/guarded entries; that is compile compatibility, not GPU
validation. Compute Sanitizer memcheck/leak-check ran all three DLR and four primitive
GoogleTests separately: 0 errors and 0 bytes leaked in both runs. `git diff --check`
passed. Logs: `/tmp/xca-bs-dlr-{cpu,cuda,library}-*.log`,
`/tmp/xca-bs-dlr-memcheck.log`, `/tmp/xca-bs-dlr-primitives-memcheck.log`.
No broader full-suite qualification is claimed for this component addition.

Next: immutable hybridization/kernel snapshots, bounded device operator caching and
streamed self-energy traversal, with complex backward special-vertex coefficient
transposition, fermionic parity and the per-path magnitude cutoff. Then both
correlator traversals, complete memory admission, consolidated bindings/MPI integration
and the remaining benchmark/optimization experiments. GH200 remains the target;
all current device checks use the local RTX A4000 development proxy.

## 2026-10-14 — E2 bounded CUDA self-energy reference

Code commit: `51b69d1` (`[bs] add CUDA self-energy traversal`). The persistent
implementation goal remains active. Self-energy is now implemented; SPGF and arbitrary
one-time correlators, Python/MPI integration and optimization experiments remain.

Added `block_sparse/cuda/diagram_evaluator.*`. It owns immutable OperatorModel,
mesh/hybridization/kernel snapshots, shared device execution and DLR objects.
Full/single/range self-energy APIs accept Fock-basis block_gf inputs and stream the
qualified CPU-order path descriptors. All numerical vertex applications, rectangular
DLR solves/convolutions, rank weights and output additions execute on the device.
Both branch directions preserve barred operators, backward special-vertex coefficient
transposition, fermionic parity, kernel multiplication order and separate prefactor/
diagram-sign scaling. Per-call input-dependent masks are recomputed from finite inputs.

One cached CSR/plain-transpose pair limits operator residency. The special vertex
recomputes each kappa product inside the mu loop and accumulates in CPU order, instead
of retaining q padded tensors. This is a correctness reference with additional upload/
launch/recomputation costs, not a selected optimization. An exact bounded host scan
of device T_out preserves the CPU's >1e-16 per-path cutoff; its 4096-complex chunk
buffer replaces full-path download. No CPU matrix replay or post-submission fallback
exists. Inputs/outputs are synchronous, the owning device is checked, and class
cleanup selects the construction device before releasing owned device members.

Admission precedes propagator/output upload: checked full input/output capacities,
seven live user-work tiles, DLR tables/RHS/five convolution tiles, kernel weights and
a configurable one-entry sparse-workspace reserve. Actual allocations use rectangular
extents; maximum dimensions are used only for this conservative bound. Defaults are
1 GiB total explicit storage and 32 MiB sparse storage, both constructor parameters;
64 MiB sparse configuration is tested. This avoids fixing target-hardware capacity
to the workstation's defaults. A global free-memory check adds 64 MiB headroom.
Flat-domain/proposal arithmetic and the reference's INT_MAX BLAS-vector limit reject
before work. Invalid counts, index ranges, shapes/meshes, nonfinite G/poles and fixed-
table budgets reject. Complete opaque-library/host peak qualification is still pending.

Memory probes sample global device free-memory drops at live phases. These include
other processes and are not process RSS or a continuous peak trace. Isolated runs
with XCA_BS_CHECK_MEMORY_PEAK=1 compare samples with the declared bound plus headroom;
ordinary CTest runs omit that assertion because other GPU processes can interfere.

Three GoogleTests now cover scalar occupation, total-number and mixed partitions,
complex ordinary/retarded coefficients, non-Hermitian hopping, both branch directions,
changed/restored/all-zero masks, empty tails, invalid calls and recovery. Selected
windows cover two third-order topologies and crossing orders 4/5 with only two
indices per window at those higher orders. Full NCA, ordinary full OCA and split-range
additivity are checked. An independent manual NCA reference checks ordinary channels.
A cutoff test places an entry beyond two whole scan chunks and checks the exact
threshold and complex modulus rather than the cuBLAS 1-norm surrogate.

CUDA-versus-exact-Fock CPU uses a relative 1e-13 block bound through sampled order 5.
The old rotated-operator versus exact CPU control remains 1e-13 at orders 1–3.
One newly added order-4 total-number tail sample exceeded it slightly: 5.8325e-25
absolute difference on a 5.4853e-12 signal (relative 1.063e-13), between the TWO CPU
controls, not CUDA. Exact signed-map/CAR controls independently qualify the direct
fields; bare rotated matrices differ within the checked 4e-14 bound. Replaying the
old rotated quartet in the same CPU constructor reproduces the old result bitwise,
isolating operator-input round-trip noise from traversal/device error. Newly sampled
old/new CPU orders 4/5 characterize amplification with a 1e-12 bound, while the
CUDA/exact-input bound stays 1e-13. This does not qualify arbitrary high-order
cancellation or change the production CPU path. The plan records this limitation.

Verification: full CPU build and 74/74 CTest entries passed in 49.66 s; full CUDA build
and 82/82 passed in 180.85 s. These runs overlapped other verification builds and are
not throughput measurements. The final configurable-budget constructor addition was
then checked in the focused library-only CUDA test (4.78 s, XCA_CUDA_SMALL=OFF) and
isolated Compute Sanitizer run: all three evaluator GoogleTests passed, 0 errors and
0 bytes leaked, with memory-bound assertions enabled. Default arithmetic is unchanged
by that option. Earlier focused normal-CUDA control passed in 4.69 s. `git diff --check`
passed. Logs: `/tmp/xca-bs-se-{cpu,cuda,library}-*.log`,
`/tmp/xca-bs-se-cuda-full-{build,test}.log`, `/tmp/xca-bs-se-final-build.log`,
`/tmp/xca-bs-se-memcheck.log`; diagnostics may disappear from /tmp.

Next: streamed correlator path/fragment descriptors and both device traversals,
endpoint sparse products/traces, SPGF and one-time correlator APIs. Complete E2 before
broad fusion or production backend registration; then phase benchmarks, dispatch/
batching/memory experiments, consolidated CUDA bindings and MPI qualification.
Large-sector operator-workspace reserves need traced target measurements; no full
N=14 evaluation, GH200 measurement, supervisor change, install or push was performed.

## 2026-10-14 — E2 correlator fragment and endpoint metadata

Code commit: `0c9e676` (`[bs] trace correlator fragments`). The full implementation
goal remains active; this checkpoint prepares device correlator execution rather
than claiming SPGF/one-time correlators are implemented on CUDA.

Extended the portable path module with forward-order descriptors for the [tau,0]
right fragment and [beta,tau] left fragment, and streaming endpoint proposals in
CPU mu/root/kappa order. Metadata records each root/end, intermediate G blocks,
spatial dimensions, special vertex, index and live-mask snapshot. There is no
numerical cutoff in this endpoint scheduler. Counts include rejected proposals;
null operators, wrong model dimensions, malformed fragment metadata, noninjective
core fragment maps, stale masks and mismatched backbones reject before visitation.
The two root tables have a checked descriptor-slot limit (65536 by default), so
N=14 scalar partitions fit without retaining all indexed evaluations at once.

Important compatibility finding: CPU correlator walks do not mask the STARTING
root before internal vertices. BlockDiagOpFun::get_block returns its stored values
even when the whole-block mask marks it zero. A two-mode scalar model with inner
c_0 and external c_1/c_1^dagger has a valid OCA SPGF endpoint that begins at the
masked fully occupied root, passes through live sectors and closes back to it.
Scaling that starting G by 1e-18 still produces a nonzero CPU SPGF term; replacing
it with zero removes that term exactly. This independent CPU regression confirms
that the device correlator must upload/preserve starting values of masked blocks,
unlike self-energy's initial post-operator pruning. Empty walks also retain roots.
The plan now records both rules explicitly.

Memory finding: all core maps and their compositions are injective. End blocks in
each side are therefore unique, and sum_root d_root*d_end <= sum_b d_b^2 by Cauchy–
Schwarz. Exact rectangular stored fragments need at most one all-sector rank function
per side. This excludes input, endpoint products, rank scratch and library overhead.
The collector validates end uniqueness, and number-sector tests check actual side
capacities over several indices/topologies, including empty-left/empty-right walks.
This replaces the interpretation that two nblocks*Nmax^2 side allocations are needed.

Three host GoogleTests cover independently known scalar NCA endpoint counts/bit
maps, masked empty walks, stale masks, bounds/invalid metadata, the numerical CPU
tiny-root case, and non-square number-sector fragments with one-function capacity.
These are host controls in both environments. Focused CPU CTest passed 2/2 entries
(new collector plus previous self-energy path regression, 0.09 s). Focused CUDA
CTest passed 3/3 entries (those two plus the actual GPU self-energy controls, 5.08 s).
No new device arithmetic was added in this checkpoint. `git diff --check` passed.
Logs: `/tmp/xca-bs-corr-path-{cpu,cuda}-{configure,build,test}.log` and the CPU
regression build log. No full-suite requalification is claimed for this host module
extension; preceding 74/82 suite results remain recorded separately.

Next: actual right-fragment left products, reverse left-fragment right products,
DLR reflection, sparse endpoints and device trace/reduction; SPGF plus general
one-time correlator full/single/range calls. Keep masked starting data, endpoint
summation order and borrowed-operator cache lifetime safe. Then complete E2 admission
and numerical gates before production bindings, MPI integration and optimization.
No N=14 diagonalization, target GPU run, supervisor change, install or push occurred.

## 2026-10-14 — E2 device SPGF and one-time correlators

Code commit: `47f41c7` (`[bs] add CUDA correlator evaluation`). All three observable
families now have C++ device reference paths. The implementation goal remains active:
production bindings/MPI, complete capacity qualification and optimization gates remain.

Added SPGF full/single/range calls, low-level sparse-endpoint correlator ranges and
many-body-operator one-time correlator full/single/range overloads for real/complex
atom_diag. Many-body endpoints are exported directly in Fock coordinates and their
shared fermion parity is inferred/validated. Internal fields still use immutable
model snapshots and the existing supported dynamic-operator domain.

The right fragment uses left products and forward edge convolutions; the left
fragment walks in reverse with sparse right products, opposite vertex kernel signs,
right edge convolutions and device DLR reflection. Both empty-side cases are covered.
Endpoint products form left * (O_mu * right) * O_kap, then nonconjugating device
traces. Traces use the owned cuBLAS handle in device-result pointer mode, restoring
host-scalar mode even on exceptions. Strided trace accumulation preserves mu/root/
kappa order. An index-local tensor receives external fermion parity before ordered
addition to the global result. Correlators have no self-energy magnitude cutoff.

Every stored starting G block is uploaded, including masked tiny values; endpoint
and post-internal-operator masks retain CPU semantics. A targeted scalar regression
isolates a 1e-18 starting block's SPGF contribution and matches the exact CPU signal
at relative 1e-13. Setting that block to zero removes the component exactly. Actual
rectangular side functions replace padded side arrays. Conservative admission adds
one all-sector function capacity to the SE bound, two correlator tensors and trace
storage. It remains deliberately conservative and requires schedule-sensitive
large-sector refinement; no practical N=14 capacity claim is made.

Temporary endpoint snapshots must not leave address-based cache keys behind. A
scope guard clears the one-entry sparse cache at correlation entry and exit, including
exceptions; changed endpoints in reused temporary allocations are checked. Input,
endpoint metadata, range/proposal arithmetic, statistics, shape and budget failures
propagate without post-submission CPU replay. SPGF results exclude dynamic flavours
from external legs. Empty ranges and low-level empty endpoint lists return shaped
zero arrays, with current input/index validation.

Numerical finding and investigation: one order-4 bosonic number-sector sample has
CPU/CUDA difference 1.18587e-21 on a 1.71186e-9 signal (relative 6.93e-13), with identical
exact Fock inputs. Changing only final trace reduction to CPU order left the difference
essentially unchanged; serial narrow RHS solves reduced it to 7.1297e-22 but did not
pass the original 1e-13 bound. Both temporary production probes were restored.
The reusable arithmetic diagnostic in test_utils/correlator_precision.hpp uses
independent long-double scalar products, triangular solves, rank transforms and
endpoint traces with the SAME cached double LU/rank/kernel constants. It shares
qualified connectivity metadata, so it is an arithmetic control, not an independent
path/physical-DLR reference. A small well-conditioned cached-LU/convolution test
checks its transpose/pivot convention independently against cppdlr.

For that sample, the arithmetic control measures CPU error 8.22557e-22 (relative
4.81e-13) and CUDA error 3.86884e-22 (relative 2.26e-13). This demonstrates a rounding
floor in the CPU control as well. Existing order-1/2/3 controls retain relative 1e-13;
added order-4 bosonic comparisons use 1e-12 against both CPU and long-double arithmetic.
SPGF samples still use 1e-13. No general high-order/cancellation guarantee follows;
qualification must grow with later experiment shapes/orders. Production CPU behavior
and the default GPU solve/trace algorithms were not changed to fit the failing sample.
The plan now records the evidence and explicit acceptance scope.

Eight evaluator GoogleTests cover self-energy plus SPGF and bosonic/non-Hermitian
correlators in scalar, number and mixed partitions, retarded channels, both branch
traversals, empty sides, changed masks, split-range additivity, endpoint cache lifetime,
real atom_diag/fermionic overloads, statistics/dimension/budget/range errors and recovery.
Five device primitive tests include complex trace values, strided slots, no conjugation,
pointer-mode restoration, alias/slot rejection and recovery.

Verification: full CPU build and 75/75 CTest entries passed in 49.23 s; full CUDA build
and 83/83 passed in 119.66 s. Final library-only CUDA focused tests passed 2/2 in 7.82 s
with XCA_CUDA_SMALL=OFF. Isolated Compute Sanitizer memcheck/leak-check ran all eight
evaluator tests with memory-bound assertions enabled and all five primitive tests:
0 errors and 0 bytes leaked in both runs. Normal focused CUDA entries passed in
7.80 s. `git diff --check` passed; unrelated user files are preserved.
Logs: `/tmp/xca-bs-correlator-{cpu,cuda}-full-{build,test}.log`,
`/tmp/xca-bs-correlator-library-final-{build,test}.log`,
`/tmp/xca-bs-correlator-{memcheck,primitives-memcheck}.log`. Investigation logs:
`/tmp/xca-bs-{trace,rhs,precision}-probe-{build,test}.log`; /tmp files may disappear.

Next: constructors/registration usable through consolidated CUDA bindings, mutable
Python block-sparse GPU selection and MPI numerical/error checks; benchmark complete
calls and phase timings, refine host/device admission, then controlled dispatch/
batching/layout/reuse experiments while preserving this reference. Target remains
GH200-class hardware. No N=14 diagonalization/full evaluation, target GPU timing,
quota-supervisor change, install or push was performed.

## 2026-10-14 — Consolidated bindings and Python/MPI integration

Code commit: `12981ad` (`[solv] block-sparse CUDA integration`). The implementation
goal remains active. The reference is now usable from Solver, while complete-call
benchmarks, capacity refinement and E3–E9 strategy experiments remain outstanding.

Registered BlockSparseCudaDiagramEvaluator in the existing _triqs_xca_cuda module
through a host-facing facade. It builds exact sparse models from real/complex
atom_diag and supports ordinary/retarded constructors, full/single/range SE and
SPGF, one-time correlator ranges and storage-bound queries. Constructor kwargs
expose device_budget and sparse_budget. No extra CUDA Python module was introduced.
Clair generated the combined schema; the existing optional-small-class guard script
worked with changed class aliases, including XCA_CUDA_SMALL=OFF. Only consolidated
CUDA wrappers were regenerated. CPU imports remain lazy and do not load CUDA.

Solver gains block_sparse_backend ('cpu' default, 'cuda' opt-in),
block_sparse_workspace_budget (1 GiB default) and block_sparse_operator_budget
(32 MiB default). Dense GPU configuration remains separate and still requires
conserved_operators=[]. Block-sparse CUDA works with automatic or explicit atomic
symmetries; contradictory family selection and invalid budgets reject. gpu.py owns
backend loading, process device selection and collective error handling. Configured
budgets go to the native constructor. Solver archives/deepcopy retain preferences;
old archives without these fields receive CPU/default budgets and no live evaluator.

GPU-active selection now follows the actual evaluator family. Collective Solver
initialization and range calls cover block-sparse as well as dense GPU evaluation.
Index-count queries are also guarded BEFORE range scattering/evaluation: a rank-local
failure there otherwise leaves a peer entering later collectives. Standalone native
and Solver initialization/full-topology calls remain local. No CPU replay follows
submitted device work; recoverable test failures create/retry the requested backend.

The new Python test uses scalar occupation, number and mixed partitions, complex
hybridization, ordinary/retarded models, all observables, self-consistency and real/
complex atom_diag binding factories. It verifies module identity, scalar/range
selectors (no Python index-vector API), archive/deepcopy/default compatibility,
unavailable CPU-build errors and invalid preferences. Two-rank runs compare
MPI-split sums with complete CPU calls. Deliberate rank-zero failures during setup,
index counting and compute admission raise on every rank; barriers and successful
retries prove that peers leave those phases rather than hanging before reductions.

Verification: full CPU build plus 76/76 CTest entries passed in 49.17 s; full CUDA
build plus 84/84 passed in 185.54 s. Focused CUDA Python tests passed 2/2 in 9.73 s;
CPU import/archive tests passed 4/4 in 1.40 s. Library-only CUDA module built and its
Python tests passed 2/2 in 32.45 s while overlapping MPI verification. Timing under
concurrent GPU tests is not a performance benchmark. MPI ran all four new Python
tests on each of two ranks successfully (43.6 s). The initially attempted executable
name 'python' was absent; no tests ran in that attempt. Successful launch used:

```bash
source ~/env/xca-cuda.sh
export PYTHONPATH="$PWD/cbuild-cuda/python:$PWD/cbuild-cuda/deps/adapol_src/src:$PWD/cbuild-cuda/deps/pyed_src:$PYTHONPATH"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpiexec -n 2 /usr/bin/python3 test/python/block_sparse_cuda.py
```

Logs: `/tmp/xca-bs-python-{cpu,cuda}-full-{build,test}.log`,
`/tmp/xca-bs-python-{cpu,cuda,library}-test.log`,
`/tmp/xca-bs-python-library-build.log`, `/tmp/xca-bs-python-mpi-test.log`,
`/tmp/xca-bs-binding-{generate,build}.log`. `git diff --check` passed. User-facing API
notes now show opt-in selection, budgets, device ownership and target-hardware limits;
documentation was not built. Unrelated user work is preserved.

Next: reproducible full-call/phase benchmarks against CPU and this retained GPU
reference, schedule-sensitive live admission for larger sectors, bounded N=10 work
and analytic/primitive N=14 proxies. Then evaluate signed maps versus sparse/dense
libraries, rank/RHS grouping, layouts, dispatch, batching and topology reuse/graphs
in separate experimental folders. Keep all dense baselines/experiments. GH200-class
hardware remains the target; no target run or N=14 diagonalization/full evaluation
was performed. No supervisor reconfiguration, install, history squash or push.


### 2026-10-14 — Bounded complete-call baseline

Goal remains active. Added shared complete-call fixtures/timers and separate CPU /
CUDA benchmark factories, with independent CMake targets under benchmarks. Generic
methodology lives at the respective roots; raw results and paired comparisons use
thinkstationP620_RTXA4000 subdirectories. Existing dense experiments are preserved.
Code commit: d79e8d7 [bench] time block-sparse reference calls. Measured results:
0944108 [bench] record block-sparse call baseline. This checkpoint is a separate
plan/work-log commit.

The CLI bounds actual modes to 2–10, orders to 1–10, index-window size to 1–8 and
samples to 1–5. It supports explicit crossing/chain topologies, all observables and
ordinary/retarded models, with physical Fock-basis propagators. It separates model
preparation, evaluator construction and complete synchronous API calls; flattening
and CSV output are outside call timing. CUDA first checks unchanged CPU results;
finite checks and exact deterministic replay are required. Screening uses
1e-10*max(CPU norm, GPU norm)+1e-14 and reports observed errors, rather than changing
strict evaluator test tolerances. CPU rows have no independent comparison and
zero error fields by definition. Device free-memory samples are not complete peak
qualification. CPU numerical-control timing includes flattening and is not the
paired baseline; the separate CPU frontend supplies the baseline.

Recorder validates folder names and positive finite timeout, runs CPU/GPU cases
sequentially, checkpoints raw rows/manifests after each process, retains failures
and refuses overwriting campaigns. Provenance pins source/binary/header hashes,
compiler, git revision/status, TRIQS prefix, GPU/CPU metadata and exact commands.
It retains preexisting user changes in the manifest rather than pretending the
entire tree was clean. No unrelated files were staged.

Verification: CPU-only configure/build and a dynamic SPGF two-index/two-sample
smoke passed; CUDA paired frontends built and a dynamic chain correlator smoke
passed. Invalid oversized count and NaN timeout rejected. Recorder syntax checked
without generating repository bytecode. The isolated reference-01 campaign completed
all 30 processes, 15 paired workloads and 82 samples successfully. All CUDA screening
checks and deterministic repetitions passed. No production evaluator was changed,
so the previous complete CPU/CUDA/MPI qualification remains the core checkpoint.

Results: small four-mode calls take roughly 1.2–1.8 ms on CPU and 18.5–56.3 ms on
CUDA. The zero-signal six-mode occupation order-three window is retained (0 accepted
paths, 0.74/1.96 ms CPU/CUDA). Eight-mode number sectors are near parity at 244.83 /
265.30 ms. Ten-mode number sectors (maximum 252) take 7008.67 / 1142.76 ms, but
that case has only one measured repetition and the CPU uses unchanged convolution.
Ten-mode scalar blocks take 1021.28 / 5603.63 ms. Relative errors for the two nonzero
six-mode higher-order controls are 3.25e-13 and 2.43e-13; remaining nonzero cases
are below 2e-14. These pass the declared screening gate, not an assertion of a
universal 1e-13 high-order bound. No target-hardware result or full Solver speedup
is inferred from selected indexed evaluations.

Next: isolate E3 operator/format/packing costs and establish the E5 hoisted CPU
convolution control, then test bounded ready-work batching/fusion for scalar and
mixed regimes. Refine schedule-sensitive capacity before larger correlators and
N=14 proxies. Keep the current CUDA reference and unchanged CPU controls available;
GH200 remains the performance target. Plan updated with observations and pending
gates. No install, push, supervisor change or N=14 diagonalization was performed.

Logs: /tmp/xca-bs-call-{cpu-configure,cpu-build,final-build,campaign}.log,
/tmp/xca-bs-call-{cpu-smoke,smoke}.csv. Persistent campaign artifacts are in
benchmarks/block_sparse_common/thinkstationP620_RTXA4000/reference-01. Temporary
logs may disappear. git diff --check passed.


### 2026-10-14 — E5 matched CPU convolution controls

Previous goal turn made verified progress (paired reference benchmarks and committed
checkpoint). Goal remains active. This turn establishes the requested matched CPU
control before interpreting device speedups; production evaluator code is unchanged.

Added experiments/block_sparse_cuda/dlr_hoisted. Hoisted time-ordered rank-three
convolution uses the installed cppdlr helper formulas and NDA products/addition
order, moving only the two complete-input Hilbert transforms out of the node loop.
It preserves the cached-LU solves. The executable compares coefficient convolution
and a full edge on identical complex inputs, requiring bitwise CPU agreement and
exact replays. CUDA uses the retained DlrOps, includes synchronization/output copy,
excludes initial input upload/setup and retains scratch. Reported device payload
excludes opaque allocations. Workload dimensions cap at 252, with rank/shape/sample
metadata and finite-output screening. Initial compilation used the wrong matmul
namespace; corrected to cppdlr::matmul before qualification/commit.

Primitive code d8f3c44, results c27011c. CPU-only/CUDA builds passed, including a
second-rank rectangular CPU smoke; invalid dimension 253 rejected. Eight isolated
cases, 108 rows passed: scalar/square/rectangular and 210/252/210 geometries plus
Lambda=10/120 controls. CPU bitwise errors are zero throughout. CUDA relative errors
are at most 1.56e-14. Largest edge original/hoisted/CUDA is 894.09/209.56/56.49 ms,
with one repetition there. Small edges strongly favor hoisted CPU over reference
CUDA, so primitive improvements alone cannot establish a device advantage.

Added a private generated full CPU evaluator control (79cb24c). CMake derives it
from current production cpp/hpp, renames the class and changes only the rank-ops
member to a cached ImtimeControl adapter. Both existing convolve calls use hoisting;
all inherited solves/transforms and path/vertex/accumulation code remain unchanged.
Generation fails if the expected member or two convolution-site patterns change,
and depends on production sources. Generated files remain in the build tree; no
installed-header override or production backend mutation. The complete-call frontend
uses shared physical workloads and requires bitwise agreement against unchanged CPU.
The extra helper tables are charged to constructor time. Both CPU-only/CUDA-linked
builds passed; CPU-only dynamic SPGF and a dynamic chain correlator with reverse
traversal/two-index range matched bit for bit. Recorder now hashes experiment source
and generation rules as well as core/fixture sources (6446c2b).

The first paired matched campaign overlapped CPU verification compilation. This was
an avoidable timing mistake, identified and corrected before drawing conclusions.
All 30 processes passed numerical checks, but hoisted-cpu-01 timing rows are explicitly
excluded in its README/manifest and retained as evidence. Verified CPU build and
first-campaign handles terminal before running clean hoisted-cpu-02; no concurrent
build or benchmark was started during that repeat. Clean campaign also passed all
30 processes, 82 samples, and bitwise CPU agreement on all 15 original physical
workloads, including zero-survival and ordinary/retarded/all-observable regimes.
Results committed separately (13979c7; hardware-link correction follows).

Clean full-call medians: eight-mode number sectors 131.30 ms hoisted CPU / 266.79 ms
CUDA; ten-mode number 5039.57 / 1130.46 ms; ten-mode scalar sectors 794.69 / 5614.41 ms.
Thus a useful large-sector device advantage survives the matched CPU comparison,
but eight-mode parity against the old CPU was misleading. These are selected first
indexed evaluations, not full Solver/topology sums. Ten-mode cases have one timed
repetition; no promotion/noise gate, target threshold or N=14 qualification follows.
Unchanged CPU results from reference-01 are separate-run context, not a substituted
same-process timing. CPU-control fields include flattening as documented.

No production changes warrant repeating the complete solver suites here; prior
full CPU/CUDA/MPI qualification is preserved, with meaningful experiment numerical
checks covering the new code. Plan updated with E5 evidence and remaining gates.
Next: E3 operator maps/sparse/dense inclusive costs (including bounded 3003/3432
proxies), phase profiling and bounded ready-work batching/fusion. E5 RHS batching,
cache keys/changed-G, threaded CPU and GH200 measurements remain outstanding.
Keep reference and all experiments; goal is not complete. No install, push or
supervisor change. Unrelated user changes preserved; git diff --check passed.

Temporary logs: /tmp/xca-bs-hoisted-{build,cpu-build,smoke,cpu-smoke,campaign}.*,
/tmp/xca-bs-hoisted-evaluator-{build,cpu-build,smoke,cpu-smoke,campaign,clean-campaign}.*.
Persistent evidence lives in experiments/block_sparse_cuda/dlr_hoisted/thinkstationP620_RTXA4000
and benchmarks/block_sparse_common/thinkstationP620_RTXA4000/hoisted-cpu-{01,02}.


### 2026-10-14 — E3 ordered operator controls

Previous goal turn made progress by establishing primitive and full-evaluator
matched CPU convolution controls. This turn adds isolated native sparse-operator
experiments; goal remains active and production evaluator dispatch is unchanged.

Code 9561926 [bench] compare sparse operator kernels; measured results 9258ed8
[bench] record sparse operator controls. Native/general CSR applies one output per
thread, traverses sorted row entries, uses no atomics and launches once for all DLR
nodes. CUDA FMA contraction is disabled for this numerical control. Both directions
are tested; the right snapshot is a plain transpose (adjoint then conjugated values),
not a complex adjoint. The library controls are retained DeviceSparseBlock ALG3
(per-node queries/descriptors/synchronization) and complex-double cuBLAS strided
batches with zero operator batch stride. Dense storage holds a single matrix,
without rank replication. No inference about ideal cuSPARSE performance is made
from the retained reference's descriptor/synchronization costs.

Fixtures enumerate adjacent total-number Fock sectors at half filling, with exact
parity for individual fields and complex weighted sums of all fields. Modes
2/4/6/8/10/14 cover rectangular dimensions up to exactly 3003-by-3432. Empty rows,
left/right products, one/all-field sums, RHS widths 1/16 and three repetitions with
reuse eight are recorded. No atom_diag diagonalization is needed. The N=14 cases
are bounded spatial slabs, not full sector-sized rank functions or a complete model.
Native CSR is general ordered storage; compact signed maps and implicit sums remain
additional strategies, not claims about this implementation.

CSV records separate host packing, route setup/upload, first-call and warm repeated
application time, numerical errors, nnz/forward and transpose degrees, operator
payload and retained sparse workspace. Context/input/output allocation and input
upload are outside warm timing; output copy/checks are also outside it. Sparse
workspace is reported separately and excluded from the operator-payload column,
correcting the initial smoke's inconsistent accounting. First call charges lazy
workspace/first-use costs. No complete peak or full Solver speedup is asserted.

Initial configure needed CUDA language initialization in the experiment directory
(the existing sibling's enable_language scope was insufficient); corrected locally
without altering production native kernels. NDA vector conversion from std::vector
was replaced with explicit host staging before building. Final build is warning-free
for the experiment. CPU-only configure succeeded and exposes no native operator
experiment target. CUDA small complex weighted smoke, two-mode narrow/empty-row
control and 14-mode seven-field width-four smoke passed. Invalid terms rejected.
compute-sanitizer memcheck on six modes/six fields/width 17 reported zero errors and
its numerical/replay checks passed. No production change warrants rerunning whole
solver suites; earlier full CPU/CUDA/MPI qualification remains the core checkpoint.

After all builds/sanitizer processes terminated, the sequential operator-01 campaign
completed all 24 processes and 432 rows. Native outputs bitwise matched the CPU
sparse reference throughout; cuSPARSE/dense maximum relative error was 5.332e-16.
All GPU routes repeated exactly. Code, binary and NDA-library hashes, commands,
CUDA compiler and machine metadata are persisted, including per-case outcomes.
The hardware README and paired summary retain all rows, not only the best results.
For width-16 complex all-field sums, native/retained-cuSPARSE left medians are
0.00560/0.33252 ms (N4), 0.03695/0.33738 ms (N10), 0.57248/0.78626 ms (N14).
Right medians are 0.00478/0.33704, 0.03583/0.34011 and 0.47965/0.79528 ms. The
native advantage shrinks as useful work grows; dense expansion is costly for these
large narrow-RHS sparse fields. No hardware-independent winner is declared.

Next: build a private complete-evaluator native candidate retaining library reference
and unchanged/matched CPU controls. Verify strict all-observable numerics, sparse
budget/lifecycle errors and inclusive timing before production promotion; do not
replace reference solely from primitive rates. Continue bounded batching and memory
refinement; broader degree/RHS, COO, implicit sums, GH200 measurements and N=14
capacity qualification remain pending. No install, push, supervisor change or user
work edits. Plan/work-log checkpoint is separate from code/results commits.

Temporary logs: /tmp/xca-bs-operator-{configure,build,cpu-configure,cpu-targets,invalid,
memcheck,campaign}.log and /tmp/xca-bs-operator-{smoke,tail-smoke,large-smoke}.csv.
Persistent artifacts: experiments/block_sparse_cuda/operator_apply/thinkstationP620_RTXA4000.
Git diff --check passed; unrelated .gitignore/untracked user work is preserved.


### 2026-10-14 — Ordered sparse evaluator candidate

Previous turn made verified E3 progress (432 primitive timing rows, native/library
numerics, memory check and committed checkpoint). This turn tests whether that
operator gain survives the actual evaluator. Goal remains active.

Code 1cc785c [bench] test ordered sparse evaluator; results 3a876e9 [bench] record
ordered evaluator timings. The private generated class reuses current reference
CPP/HPP, renames its class and replaces exactly three DeviceSparseBlock type
references. CMake checks that pattern and regenerates on reference source changes.
All traversal/mask/cutoff, DLR, special-vertex, endpoint invalidation and ordered
accumulation code remains unchanged. Installed headers, core implementations and
Python bindings are untouched; reference/control tests and classes remain available.

OrderedSparseBlock owns forward/plain-transpose CSR and its shared DeviceOps owner.
It checks exact explicit payload against budget before allocation/upload, validates
C-layout/shapes/nonoverlap/pointer span/grid bounds/device ownership, and finishes
each apply before caller temporaries may be destroyed. This synchronization is
necessary even though the primitive launch API is asynchronous. Cleanup selects the
construction device, destroys arrays/shared owner, then restores the caller. The
local machine has one GPU: this does not establish the switch-device branch. General
CSR is retained (not yet a compact map/implicit-sum representation), with no library
workspace. Zero-entry snapshots write zeros on both left/right products.

The isolated executable runs all eight existing CUDA evaluator tests against the
private class, plus two focused snapshot tests. Cases include exact-input SE/SPGF,
bosonic/fermionic endpoints, retarded/mixed/scalar partitions, manual NCA, changed
masks, weak starting values, arithmetic controls, range additivity, budget/errors,
input metadata ownership, alias/shape rejection and shared-owner lifetime. Snapshot
budget 199 bytes rejects before upload where 200 bytes is the exact payload.
Candidate construction/application throws are not replayed on CPU/library backend.
Full 10/10 GoogleTests passed (CTest entry 1/1, 3.84 s). Existing strict test gates
were unchanged. Focused memcheck rerun after span/zero-right improvements passed
four tests with 0 errors and 0 bytes leaked in 0 allocations. CUDA frontends built;
CPU-only configure and target listing confirm the candidate is omitted there.
No full solver suite was rerun because production code/bindings are unchanged.

Recorder now also hashes native .cu source, covering candidate kernel provenance.
After all build/test/sanitizer processes terminated, the sequential paired
ordered-evaluator-01 campaign completed 30 processes / 82 rows. All 15 original
physical workloads include ordinary/retarded SE/SPGF/correlators and the zero-survival
window. CPU screening and exact GPU replays passed; matched CPU still agrees bitwise
with unchanged CPU. Proposed/accepted/pruned counts exactly match retained reference
for each case. Maximum candidate screening relative error is 3.645e-13 in a weak
higher-order contribution, not a universal 1e-13 claim. Source/native kernel/binary/
header hashes, commands and machine metadata are persisted. Counts/latencies include
complete calls; packing/transfer/temporary costs remain present in candidate timing.

Candidate versus prior clean library reference medians: eight-mode number sectors
85.67 versus 266.79 ms; paired matched CPU is 135.24 ms. Ten-mode number sectors
690.85 versus 1130.46 ms, paired CPU 5064.84 ms. Ten-mode scalar sectors improve
from 5614.41 to 2710.87 ms but still trail paired CPU 788.10 ms. Modes 10 have one
repetition and cannot satisfy a noise-qualified promotion gate. Most small cases
have three samples and show substantial complete-call gains; the zero-survival
window retains overhead. Library values are from a separate clean run, disclosed
in summaries. No full-topology-sum or Solver throughput is claimed.

Next: qualify an optional production native operator route, retaining library path
and no-native-compiler support. Wire explicit capability/selection through consolidated
bindings and MPI Solver; run complete CPU/CUDA/library-only/MPI checks before promotion.
Then address scalar batching/DLR amortization, changed-G/operator reuse and wider
RHS/memory bounds, and revisit topology sharing based on launch evidence. GH200
remains the target. No target run, N=14 full model, install, push or supervisor
change. Unrelated user work is preserved. Plan/work-log commit stays separate from
candidate code/results; git diff --check passed.

Logs: /tmp/xca-bs-ordered-{configure,build,test,memcheck,cpu-configure,cpu-targets,
campaign}.log. Persistent evidence: benchmarks/block_sparse_common/thinkstationP620_RTXA4000/ordered-evaluator-01
and experiments/block_sparse_cuda/operator_apply. Temporary logs may disappear.


### 2026-10-14 — Optional production ordered operator backend

Previous turn made progress by qualifying the private native evaluator and paired
complete-call evidence. Goal remains active. This turn integrates the qualified
kernel into production C++, leaving Python/MPI selection for the next integration
step rather than claiming that promotion is finished.

Code commit 0450ab1 [bs] add ordered CUDA operator backend. Added independent
XCA_CUDA_BLOCK_SPARSE AUTO/ON/OFF native compilation with CUDA-enabled NDA and
compiler checks. Kernel static library is linked/exported/installed with the core;
CUDA standard 17, PIC, FMA contraction disabled. Dense CUDA-small's control remains
independent. CUDA support itself is still inherited from NDA/TRIQS; this option
only controls xca's native sparse kernel. AUTO omits native compilation on CPU/no
compiler; an explicit unsupported ON errors during configure. No installed prefix
was changed in this work.

Public C++ SparseBackend has Library/Ordered, with ordered_sparse_available() as a
compiled-capability query. Evaluator and snapshot constructors accept explicit
selection; existing calls use Library defaults. Invalid enums and unavailable
Ordered requests reject. The reference's owned CSR/plain-transpose arrays, shared
owner, device/contiguous/overlap checks, checked budgets and cleanup are reused.
Native applies launch over all nodes then finish before callers release temporaries.
All-zero blocks use the existing zero operation. Retained workspace stays zero for
native applications. CSR descriptors are still constructed with snapshots; production
constructor cost can differ from the private candidate and must be timed directly.
No post-submission replay/fallback or performance threshold was introduced.

Added a production native test adapter to run the same eight physics/API tests as
the library reference. A separate capability/budget test covers invalid enums,
missing native compilation, exact 200-byte payload versus 199-byte rejection,
shape errors, native CPU comparison and all-zero output. Native test target initially
collided with the private experimental library name; corrected to a distinct name
before building/testing. The isolated prototype snapshot accepts the extra internal
selection argument so its original experiment remains buildable and available.

Verification: native-enabled CUDA configure/build passed; three focused CTest
entries passed (library evaluator, native evaluator, capability/budget), 11.67 s.
These timings overlapped independent build verification and are not benchmark data.
Native evaluator memory-check/leak-check on two endpoint/cache error/recovery and
masked-starting-value cases passed with 0 errors, 0 bytes leaked. Library-only CUDA
configured with XCA_CUDA_SMALL=OFF and XCA_CUDA_BLOCK_SPARSE=OFF, built both
library/capability executables, and passed 2/2 CTest entries in 7.78 s; its native
request rejection is exercised. CPU configure/build of the new guarded test passed;
its CTest entry passes with the no-CUDA case disabled (not GPU qualification).
The private candidate rebuilt and passed its ten-test CTest entry in 3.83 s after
production constructor changes. git diff --check passed; unrelated work preserved.

Next: extend consolidated facade/bindings and Solver preferences with explicit
library/ordered operator selection and archive defaults, including early capability
rejection. Rebuild complete CPU/CUDA/library-only Python modules and run full suites
plus two-rank collective failure/recovery checks for native selection. Measure the
actual production route with matched controls, then progress scalar ready-work
batching/DLR amortization, wider RHS/peak bounds and target qualification. Existing
reference and all experiments remain. No Python/MPI qualification, production
performance result, GH200 run or N=14 full model is asserted here. No install, push
or supervisor changes. This plan/work-log checkpoint is separate from code.

Temporary logs: /tmp/xca-bs-native-{configure,build,test,memcheck,cpu-configure,
cpu-build,cpu-test,library-configure,library-build,library-test,experiment-build,
experiment-test}.log. Build/test artifacts in the three retained cbuild trees.


### 2026-10-14 — Native selection through bindings and MPI

Previous goal turn made verified progress (optional production C++ backend plus
native/library/no-native checks). Goal remains active. This turn integrates the
native route through the consolidated bindings, Solver and archives and measures
the production implementation itself.

938e73b [solv] select ordered sparse CUDA operators: consolidated facade constructors
accept operator_backend='library'/'ordered'. Their private parser checks capability
before export_model, so unsupported native requests do not build a sparse model or
submit GPU work. Module block_sparse_ordered_available exposes compiled capability;
gpu.py owns lazy capability/selection. Solver gains block_sparse_operator_backend,
validated/stored/archived with library default. Old archive dictionaries restore
library defaults, deepcopy/HDF preserve ordered preference, and CPU evaluation can
keep preferences without importing CUDA. Selection occurs before device configuration
in GPU setup; existing collective setup/count/compute guards apply to both routes.
No new CUDA Python module or index-vector API was introduced.

Regenerated only the consolidated CUDA wrapper with clair and the optional-small
class guard. The first two focused runs failed because the new capability function
was outside the module's explicit TOML export regex. Corrected the regex and verified
the generated function entry before rebuilding; both final module variants pass.
This was an export configuration error, not a numerical failure. Constructor
real/complex ordinary/retarded template declarations/instantiations include the new
string argument. Native-disabled wrappers retain the full schema and report false
capability, rather than removing public classes or silently selecting another route.

Python tests now run all observables on library and native across scalar, number
and mixed partitions with/without retarded channels. They cover archive/deepcopy/
old defaults, range-only selectors, native-disabled direct facade and Solver errors,
collective rank-local invalid selection, initialization/count/compute failures,
barriers/recovery and self-consistency. Native error tests reuse the same meaningful
Solver workflows as library controls. Five unittest methods run in each eligible
CUDA build; CPU skips hardware cases while exercising preference/archive behavior.

Final full builds passed in CPU, CUDA and native-disabled CUDA trees. Complete CPU
suite passed 77/77 in 49.79 s; complete CUDA 87/87 in 275.78 s; native-disabled CUDA
77/77 in 122.26 s. Focused regenerated-module tests passed for native and library-only
(70.56/66.74 s while concurrent verification was running). Two-rank MPI passed all
five Python tests on each rank in 109.1 s, including native numerical results and
failure/recovery. Verification GPU jobs overlapped; those durations are not timing
benchmarks. The same absolute /usr/bin/python3 and build/dependency PYTHONPATH setup
as the previous MPI checkpoint was used. Documentation API selection notes updated;
documentation was not built. No installed prefix was changed.

5d0a825 [bench] time production ordered evaluator adds a separate explicit native
frontend (actual production class) with unchanged shared fixtures/call scope. The
recorder now hashes production kernel source/CMake as well as the experiment sources.
After all verification/build/MPI handles were terminal, ordered-production-01 ran
30 isolated processes / 82 samples successfully. Results 008e4e5 [bench] record
production ordered timings. Counts match retained library reference; matched CPU
outputs remain bitwise original CPU; GPU screening/replay checks passed. Maximum
relative screening error remains 3.645e-13 for a weak higher-order contribution;
strict C++ gates were not loosened. All timing rows/provenance are retained.

Production native / prior clean library / freshly paired matched CPU: N8 number
92.88/266.79/130.20 ms; N10 number 698.03/1130.46/5091.18 ms; N10 scalar
2845.91/5614.41/802.02 ms. The last two have one repetition, not noise-qualified
promotion evidence. Production retains CSR descriptor construction even for native
application, unlike the private snapshot, and its latency is measured separately.
No full Solver/topology-sum speedup or target threshold is claimed. Library stays
default. GH200 remains the performance target, with no target run/N=14 full workload
performed here. Goal is not complete: small-block batching/DLR amortization, wider
RHS, memory refinement, topology reuse decisions and target qualification remain.

Next: isolate remaining per-node/product/solve overhead and evaluate bounded rank/
RHS/work grouping or fusion without changing per-path cutoff/accumulation semantics.
Retain both production routes, unchanged/matched CPU and every experiment. No install,
push, supervisor change or unrelated user work edits. Plan/work-log commit remains
separate from code and benchmark commits; git diff --check passed.

Logs: /tmp/xca-bs-native-bindings-generate.log,
/tmp/xca-bs-native-python-{cpu,cuda,library}-{build,test,module,full-test}.log,
/tmp/xca-bs-native-python-mpi-test.log,
/tmp/xca-bs-native-production-{bench-configure,bench-build,campaign}.log.
Persistent results: benchmarks/block_sparse_common/thinkstationP620_RTXA4000/ordered-production-01.
Temporary logs may disappear; three build trees and source checkpoints are retained.


### 2026-10-14 — Packed scalar edge investigation

Previous goal turn made verified progress: native Python/MPI integration and complete
CPU/CUDA/native-disabled qualification plus actual production benchmarks. Goal stays
active. Initial proposed rank-product batching was already present in authoritative
DeviceOps::product; corrected direction before changing it. This turn investigates
packing independent scalar edges across RHS instead, as E4/E5 prescribe.

50e1f7e [bench] batch scalar DLR edges adds isolated experiments/block_sparse_cuda/scalar_edges;
9f6b1c6 [bench] record scalar edge batching stores the bounded results. Production
code and numerical policies are unchanged. Independent DlrOps does two coefficient
solves and convolution per scalar lane. Packed route uses cached LU/getrs on r-by-L
RHS, shared rank transforms and ordered complex-double elementwise diagonal/cross
products, FMA disabled. Every lane retains all nodes; no outer product/cross-lane
sum, inverse or custom LU is introduced. Packed route has retained scratch/table
storage; it currently duplicates helper tables rather than exposing production
DlrOps internals. This is a segment candidate, not a full evaluator scheduler.

Inclusive timing gathers separately owned device inputs via cudaMemcpy2DAsync,
computes and scatters to separate outputs. Resident variant excludes input gathering
but still scatters. Initial context/allocation/uploads and host downloads/checks are
outside timing. Explicit work bytes exclude common inputs/outputs and opaque library
peak; these are not full-evaluator admission numbers. CPU Hoisted supplies numerical
reference per lane; matched CPU batching timings are not yet supplied. Per-lane
screening is 1e-12*lane norm+1e-30, avoiding a batch norm hiding weak/zero results.
Exact GPU replay is required. Inputs include zero lanes and 1e-10-scale coefficient
pairs, producing outputs near 1e-20. No completed-path cutoff is applied here.

CUDA experiment build and CPU-only configure passed. Invalid lane count 257 rejects.
Initial 32-lane smoke passed; width-17 Lambda=120 memcheck plus full leak check passed
with 0 errors / 0 leaked allocations. After build/sanitizer completion, scalar-01
ran all 15 isolated cases / 135 rows successfully: lanes 1/2/4/8/17/32/64/127/256 at
Lambda40, plus 1/17/127 at Lambda10/120. Three repetitions per case. Maximum per-lane
relative error was 2.7657e-15; all outputs repeated exactly. Source/binary/cppdlr
header hashes, commands/compiler/machine metadata and raw samples are persisted.

Rank23 medians independent / packed inclusive: 32 lanes 9.683 / 0.723 ms, 127
38.584 / 1.520 ms, 256 77.673 / 2.627 ms. Single lane 0.310 / 0.279 ms retains
fixed overhead; no target threshold is selected. These compare GPU segments, not
CPU or full Solver throughput. All sampled tails and zeros are retained. No broad
solver suites repeated because this commit only adds an isolated experiment; previous
full qualification remains the production checkpoint. Plan updated with evidence.

Next: timed independent/packed CPU controls, then bounded evaluator ready-work
scheduler with copied borrowed backbone metadata, original contribution order and
per-path cutoff. Special vertices and correlator fragments require full semantics;
do not substitute an edge microbenchmark for full functionality. Continue memory/
wider RHS/topology and GH200 qualification. Native/library references and all
experiments remain. No install/push/supervisor changes or unrelated user work edits.
Plan/work-log checkpoint separate from code/results. git diff --check passed.

Logs /tmp/xca-bs-scalar-{configure,build,cpu-configure,invalid,memcheck,campaign}.log,
/tmp/xca-bs-scalar-smoke.csv. Persistent artifacts under
experiments/block_sparse_cuda/scalar_edges/thinkstationP620_RTXA4000/scalar-01.


### 2026-10-14 — Matched scalar batching and pointer tables

Previous turn made E4/E5 progress (packed scalar GPU segment with per-lane numerical
and memory evidence). Goal active. Added matched timed CPU controls before deriving
CPU speedups; their improvement materially changes the GPU-only interpretation.

2091ba8 [bench] compare matched scalar batches: CPU independent uses cached LU and
hoisted scalar convolution, packed CPU combines RHS/rank transforms and elementwise
products with inclusive/resident gather-scatter variants. CPU temporary allocations
are timed, work bytes NA. GPU pointer-table kernels replace per-lane cudaMemcpy2D
calls with single gather/scatter launches. Tables borrow addresses from buffers
owned for the whole run, uploaded during setup; no dynamic ownership guarantee is
claimed. Three CPU/five GPU routes compare identical inputs, single-thread CPU.
Bound expanded modestly to 1024 lanes to test practical larger ready cohorts.

46acc53 [bench] record matched scalar batches: all 17 isolated cases / 408 raw rows
passed per-lane CPU screening and exact replay; largest relative error 2.7657e-15.
Lane/rank/mesh tails, zero and weak lanes retained. CPU versus pointer-table GPU
rank23 medians: 127 lanes 0.202/0.471 ms (CPU wins), 512 lanes 0.852/0.489 ms,
1024 lanes 1.666/0.763 ms. Per-lane-copy GPU at 127 is 1.553 ms, showing submission
cost matters. Original GPU-only scalar-01 results preserved; no full Solver or
hardware-independent batch threshold follows from these segment ratios.

01a27a6 [bench] check scalar buffer updates adds a post-timing in-place complex
input scaling and fresh CPU references, followed by descriptor gathering and
per-lane validation. Existing addresses stay fixed; this checks fresh contents,
not reallocated pointers. Rank32/17 lanes and rank23/1024 lanes passed. After this
change, width17/Lambda120 memcheck+leak-check passed with 0 errors / 0 leaked bytes.
All original timing rows used the same initial inputs; changed-data validation runs
after measurement. Campaign provenance pins the pre-update benchmark code; the
later update checks are separate evidence, not silently retrofitted timing data.
Invalid lane count 1025 rejects. CUDA build passed; production files are unchanged,
so full solver suites were not repeated. Previous full CPU/CUDA/native-disabled/MPI
qualification remains the production checkpoint. Generic experiment README rewritten
to describe current routes/bounds; historical data stays in hardware subfolders.

Next: factor the packed scalar operations into a bounded candidate scheduler using
copied path metadata and explicit ready-operation ownership. Preserve individual
completed-path cutoff and original accumulation/fan-in, handle special mu/kappa
sums and irregular correlator fragments or retain general paths where a cohort is
unsupported. Charge discovery/descriptor refresh/setup, changed G/address cases and
peak memory in inclusive full calls before promotion. Do not substitute a large
uniform segment for evidence of real ready-work density. GH200 target qualification,
wider matrices/memory and topology sharing remain pending. Goal is not complete.
No install/push/supervisor changes; unrelated user work preserved. Plan/work-log
commit separate from code/data; git diff --check passed.

Logs: /tmp/xca-bs-scalar-cpu-control-{build,memcheck,invalid,campaign}.log,
/tmp/xca-bs-scalar-update-{build,memcheck}.log, /tmp/xca-bs-scalar-update-{test,large-test}.csv.
Persistent results: experiments/block_sparse_cuda/scalar_edges/thinkstationP620_RTXA4000/scalar-cpu-01.


### 2026-10-14 — Bounded scalar self-energy scheduler

Previous turn made progress through matched scalar CPU/GPU controls and pointer-table
packing evidence. Goal remains active. This turn builds a private complete-evaluator
candidate rather than substituting edge rates for actual diagram timings.

fd740bd [bench] schedule scalar diagram paths factors qualified ScalarPack from the
edge executable into a shared experimental header and adds scalar_paths. Generated
CPP/HPP reuses the current reference, inserting a scalar worker and replacing the
SE visitor (CMake checks the expected visitor). At most 512 owned Backbone/path
copies persist; NDA regular arrays/vectors and vertices are copied so borrowed
visitor metadata can be reused. Cohorts separate special positions and retain
original discovery order. Batch allocation is lazy for zero-survival windows.

Eligibility is all blocks dimension1 and every symmetry set singleton. This limits
the optimization cohort, not functionality: other scalar sets/rectangular models
and all correlators retain general execution. No Python/MPI candidate selector is
exposed. Scalar vertices, separate kernel multiplications, packed convolutions,
special kappa/hybridization/mu, prefactor and parity follow original order. Per-lane
completed max magnitude >1e-16 gates contributions, followed by original sequential
fan-in. Inactive tail lanes gather zeros and never accumulate. G/weights/descriptors
are rebuilt/read each current call; no stale propagator cache is introduced. All
preparation/transfers/cutoff/fan-in costs are in complete-call timing.

Qualification initially passed all eight unchanged evaluator tests, plus six-mode
640 accepted paths (eight indexed evaluations) spanning cohorts/tail and ten-mode
2304 accepted paths. The latter separately used three repetitions at about 89.25 ms
median, relative error 4.93e-15 vs CPU screening. Multi-cohort memcheck+leak-check
reported zero errors and zero leaked allocations. Generic CPU configuration passed;
production code unchanged, so full production suites were not repeated.

629d086 [bench] fix scalar workspace bound corrects a failed final diagnostic edit:
clang-format had changed indentation and the scripted helper insertion did not
match, so public bound/preflight referenced a missing member. I started a campaign
before noticing that rebuild's nonzero exit, using the prior executable; this was
an avoidable verification mistake. The campaign had already completed when termination
was attempted. scalar-path-01 is retained but every row is explicitly excluded from
revised-source qualification. Corrected helper builds, inherited tests pass again
(6.57 s), shared edge component checks pass, and corrected multi-cohort memcheck has
0 errors/leaks. Extra conservative batch storage is included consistently in both
admission before input upload and public self-energy storage bounds. Host metadata,
opaque library/granularity effects remain outside complete peak proof. Source
provenance now hashes .inc scheduling code. No unresolved broken build remains.

After corrected verification handles were terminal, scalar-path-02 ran all 30
processes / 82 rows successfully. 86380de [bench] record scalar path scheduling
stores corrected and excluded campaigns with explicit explanations. CPU screening,
exact replay and proposed/accepted/pruned parity with production native pass.
Maximum screening relative error remains 3.645e-13 on the preexisting weak higher-order
non-scalar control; scalar ten-mode error is 4.926e-15. Ten-mode scalar SE is 90.75 ms
versus prior native 2845.91 ms and paired hoisted CPU 798.37 ms; number-sector fallback
704.24 vs 698.03 ms prior native. Ten-mode campaign points have one repetition; the
separate scalar three-sample check is not silently merged into summary. This is
selected-index throughput, not full Solver or matched batched-CPU scheduler evidence.
All old implementations/experiments remain. GH200 stays the target.

Next: full batched scalar CPU control and explicit threshold-adjacent/tail/changed-G/
lifecycle/budget qualification, then larger-set and correlator ready-operation
scheduling or general-route retention with measured dispatch. Integrate/promote only
with complete CPU/CUDA/no-native/Python/MPI checks; continue wider RHS/memory/topology
and target qualification. Goal not complete. No install/push/supervisor/user work
changes. Plan/work-log commit separate from code/results; git diff --check passed.

Logs: /tmp/xca-bs-scalar-path-{configure,build,test,cpu-configure,memcheck,campaign,
clean-campaign}.log; /tmp/xca-bs-scalar-path-{tail-smoke,large-smoke,component-test}.csv.
Persistent code experiments/block_sparse_cuda/scalar_paths; corrected paired data
benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-path-02.


### 2026-10-14 — Independent scalar scheduler semantic gates

Previous turn made progress by implementing and qualifying the bounded scalar
candidate plus corrected inclusive full-call evidence. Goal remains active. This
turn adds independent promotion gates, not merely implementation-mirroring tests.
4a3b5a4 [bench] qualify scalar scheduler semantics adds four qualification cases to
the private evaluator test executable; production files/bindings are unchanged.

A six-mode scalar OCA range with 640 accepted paths crosses the 512-cohort boundary
and leaves a tail. It compares numerical results and contributing/discovery counts
against production native, then checks split-range sums. Complex in-place G scaling,
a dead block and all-zero inputs verify fresh data/masks and empty discovery.
A separate budget case uses candidate bound minus one byte, verifies rejection
with zero sampled memory delta/accepted count, permits an empty range, and recovers
through a correctly admitted instance. Multi-field retarded scalar symmetry sets
have identical connection rows but multiple mu/kappa members; the candidate reports
no extra scalar workspace and matches retained general execution.

The NCA cutoff case keeps all input blocks live and probes adjacent representable
values below/on/above the actual completed-path boundary. Each path's output must
be excluded at maxabs<=1e-16 even if adding paths would exceed that threshold, and
accepted above; numerical results/contributing counts match the native reference.
Initial assumptions caused two fixture failures, not candidate mismatches: zero-pole
hybridization is reconstructed through DLR and differed slightly from exact 0.5,
so nominal amplitude2e-16 was already above cutoff; a 1e-20 constant did not produce
the intended multi-field connection set in operator construction. The corrected
fixture derives boundary from actual reconstructed values and uses a resolvable
1e-8 constant for the grouping test, without modifying production grouping/export
behavior or tolerances. NCA excludes convolution; higher-order rounding boundary
qualification is still needed, not inferred from this case.

Final CUDA build and all 12 GoogleTests pass (CTest1/1,8.27s). Focused memcheck+full
leak check on the four new cases passes in17.9s with0errors and0bytes leaked. Existing
numerical tolerance for these exact scalar/reference comparisons is relative1e-13
plus1e-30; no production/inherited gate was loosened. No full solver suite repeated
because this turn changes only experiment qualification. Prior production full
CPU/CUDA/no-native/MPI checks and all experiment data remain available.

Next: matched full batched-scalar CPU control, higher-order cutoff sensitivity and
remaining ownership/address/irregular special/correlator gates before production
promotion. Then qualify any selector in full builds and MPI; continue wider memory,
topology sharing and GH200 work. Goal not complete. No install/push/supervisor/user
work edits. Generic candidate README and plan updated; documentation checkpoint
separate from code; git diff --check passed.

Logs: /tmp/xca-bs-scalar-qualification-{configure,build,test,memcheck}.log.
Persistent qualification source: experiments/block_sparse_cuda/scalar_paths/qualification.cpp.


### 2026-10-14 — Full-call matched scalar CPU scheduler

Previous turn made verified progress through independent scheduler semantic/memory
gates. Goal active. This turn closes the missing matched full-call CPU control,
preventing algorithmic batching gains from being labeled device acceleration.

1427e2a [bench] match scalar path CPU scheduling adds cpu_scheduler.hpp and a CPU
frontend. Scalar CPU uses the same packed RHS/cached LU and hoisted transforms as
qualified CPU edge controls. CPU and GPU now share recipe.inc for exact stage order:
vertices/edge kernels, special kappa/hybridization/mu, prefactor/parity and traversal.
Both use the same streamed structural path discovery, bounded 512 copied descriptors,
cohort splits, per-path magnitude cutoff and original fan-in order. Unsupported shapes
or multi-field sets and all correlators forward to unchanged CPU/general GPU. The
CPU control exports the same direct sparse model for scalar operations and retains
the original CPU evaluator for control/fallback; no production CPU API changed.

The CPU frontend builds/runs without CUDA in cbuild; a CPU-only scalar numerical/
replay check passed. CUDA-linked multi-cohort six-mode/eight-index and ten-mode
three-sample checks passed with zero numerical error against original CPU, including
exact replay. Those validation timings overlapped CPU build activity and were not
used as benchmark evidence. Shared recipe extraction left all twelve GPU tests
passing (8.48 s). No kernels or numerical policies changed; previous focused memory
qualification applies to the unchanged GPU stage operations. Full production suites
were not repeated for these private experiment changes.

After all verification/build handles were terminal, scalar-matched-01 completed
30 isolated processes / 82 rows. 6f6b93c [bench] record matched scalar path timings
stores hashes/commands/repetitions and hardware comparison. All CPU outputs are
bitwise original CPU in the sampled matrix, and GPU screening/exact replay pass.
CPU path columns are NA (no observer), not claimed matching counts. Other observable/
geometry rows use unchanged CPU, so comparisons are explicitly scoped to eligible
scalar self-energy. Old benchmarks and all implementations remain.

Matched full-call data changes interpretation: four-mode scalar SE 2.723 ms CPU /
3.429 ms GPU; ten-mode scalar SE 23.262 / 89.399 ms; zero-survival six-mode window
0.032 / 1.981 ms. The previous GPU-versus-hoisted-CPU speedup mostly came from
algorithmic batching. Ten-mode has one repetition; no target dispatch/promotion gate
follows. No claim that scalar GPU beats matched CPU is made. GH200 remains the target,
with no target run/N=14 full workload. The recorded GPU-losing result guides the next
experiment rather than hiding it or installing a CPU replay path.

Next: isolate serialized cutoff/per-path fan-in host-device handoffs and test bounded
output-owned GPU accumulation preserving each output's original contribution order.
Retain the serial candidate and matched CPU as controls. Repeat cutoff/tail/mask/
budget/lifetime gates and matched full calls before production/Python/MPI promotion;
continue wider memory/topology/hardware work. Goal not complete. No install/push/
supervisor/user-work changes. Plan/work-log/README update separate from code/data.
git diff --check passed.

Logs /tmp/xca-bs-scalar-matched-{configure,build,test,cpu-configure,cpu-build,campaign}.log;
/tmp/xca-bs-scalar-matched-cpu-{tail,large,only-test}.csv. Persistent paired data under
benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-matched-01.


### 2026-10-14 — Output-owned ordered GPU fan-in

Previous turn made progress by establishing full matched CPU scheduling and showing
the scalar GPU still loses. Goal active. This turn isolates serialized per-path
accumulation handoffs rather than promoting based on old-baseline speedups.

e52609d [bench] preserve ordered scalar GPU fan-in adds a conditional private variant
and separate library/frontend/test targets, retaining the serial candidate. Host
maxabs cutoff per lane stays unchanged. Surviving lanes are grouped by destination
in discovery order. Each GPU thread owns one output/rank element, loads the prior
value and sequentially adds each lane, with no atomics/tree reduction/reassociation.
Outputs may execute independently in parallel, but each output's original arithmetic
order is preserved across/cohorts. Pointer/offset/index arrays remain owned until
completion, bounded by the 512-lane cohort; existing extra admission has room for
metadata. All host grouping/upload/kernel costs enter complete calls. No production
selector or input-policy change was introduced.

Baseline twelve tests and variant thirteen tests pass (2/2 CTest entries,16.52s).
The additional adversarial test uses large cancelling complex terms, interleaved
output ownership, nonzero existing output and a second application to expose reorder
or reset bugs; GPU results match sequential CPU exactly. Five focused qualification/
order tests pass memcheck+full leak checking in17.6s with0errors/0bytes leaked.
This includes cutoff-adjacent NCA, multi-cohort/tails, changed masks/data, rejected
budgets/empty ranges and scalar multi-field general fallback. No inherited tolerance
was loosened. Production unchanged, so full solver suites were not repeated.

After all verification handles were terminal, scalar-fanin-01 completed30isolated
processes/82rows. cf28179 [bench] record ordered scalar GPU fan-in stores source/
binary/kernel/recipe hashes, commands, numerical/count comparisons and raw timings.
Screening and exact replay pass; proposed/accepted/pruned counts match retained
serial candidate. Maximum screening relative error remains3.645e-13 on the existing
weak higher-order non-scalar control; large scalar error4.926e-15. Serial values are
from a prior clean run, disclosed. Other geometries/observables remain general paths.

Ten-mode scalar selected SE now56.14ms versus prior GPU89.40ms and matched CPU
22.91ms. Separate three-sample validation measured57.36ms median; campaign's ten-mode
point has one repetition. This is a useful GPU bottleneck reduction, still not a
win over matched CPU or whole Solver throughput. GPU-losing/zero-survival rows remain
visible. No promotion/default/target threshold follows; GH200 remains performance
target. All previous implementations and experiments retained.

Next: profile residual host weight/descriptor preparation, rank-solve/transform
synchronization and cutoff transfer; use measured breakdown to choose bounded caching
or fusion with complete propagator/model/topology invalidation. Continue general
special/correlator scheduling, memory and target gates. Goal not complete. No install,
push, supervisor/user-work edits. Plan/work-log/README update separate from code/data;
git diff --check passed.

Logs: /tmp/xca-bs-fanin-{configure,build,test,memcheck,campaign}.log and
/tmp/xca-bs-fanin-large-smoke.csv. Persistent paired results under
benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-fanin-01.


### 2026-10-14 — Scoped profiling and scalar device arenas

Previous turn made progress through output-owned ordered accumulation and matched
CPU evidence. Goal active. This turn uses measured runtime categories to target the
remaining overhead instead of assuming dense arithmetic dominates scalar cases.

1e3d0c0 [bench] pack scalar device storage adds profile_fanin/profile_arena scoped
frontends and an isolated arena variant. cudaProfilerStart/Stop encloses only the
common runner's second (warmed) scalar self-energy range call; model setup, CPU
control and first GPU call are excluded. Nsight CUDA-only tracing disables CPU
sampling/context switches. The wrapper's recorded time includes profiler/export
cost and is excluded from performance tables. ba6d0c8 [bench] profile scalar device
storage records raw textual reports, version/options, exact case and binary/report
hashes. Large binary .nsys-rep artifacts remain in /tmp and may disappear, not tracked.
The baseline trace predates source adaptation; executable hashes identify each
captured binary without pretending both came from one clean source revision.

The first trace shows 3206 cudaMalloc,3208 cudaFree,1121 cudaMemcpy,2055 cudaMemcpy2D,
1253 cudaStreamSynchronize and1029 cudaMemsetAsync calls; GPU arithmetic sums about
3.7ms while allocation/copy API intervals dominate. Arena variant uses two owned
(block,r,1) tensors and scalar block views through reshape, rather than one allocation
per block. Masked input is packed as zero, output is zeroed once, and all result data
is downloaded once before owned CPU Gf construction. Host packing and those transfers
remain timed. Views live only within storage-owner scope, including exceptions;
scalar worker is templated over array/view types. General shapes/correlators keep
original execution, and payload/preflight conservatism is unchanged. Previous serial
and ordered-per-block controls remain separately built and available.

Arena/control thirteen-test suites pass 2/2 CTest entries (16.54s). Five focused
semantic/order tests pass memcheck+full leak check in17.5s with0errors/0bytes leaked.
These exercise tails, changed data/masks, adjacent NCA cutoff, budget/empty ranges,
multi-field fallback and cancellation-order preservation. No tolerance changes.
Separate three-sample ten-mode scalar smoke gives25.50ms median, error4.926e-15.
Production code/bindings unchanged; prior full production suites remain qualified.

A repeat scoped trace confirms mechanism: cudaMalloc137, cudaFree139, cudaMemcpy99,
cudaMemcpy2D8, stream synchronize230, async memset6. These trace counts/categories
are not unprofiled runtime rates; no claim of complete allocator/opaque peak proof.
After profiler/build/test handles were terminal, scalar-arena-01 ran30isolated
processes/82rows successfully. 00aa8e9 [bench] record contiguous scalar storage
preserves exact replay/numerical gates, matching discovery counts, hashes/commands
and raw rows. Scalar SE selected ten-mode latency25.33ms vs prior ordered-per-block
56.14ms and matched CPU22.99ms. Campaign ten-mode has one repetition; separate
three-sample smoke not silently merged. Other observer/geometry paths are general,
not scalar CPU batch comparisons. Maximum screening relative error remains3.645e-13
on the preexisting weak higher-order non-scalar control.

The target remains GH200, not A4000. Near CPU parity is progress, not a robust target
promotion threshold, full Solver speedup or N=14 qualification. Next use reduced
trace to examine residual host weight/descriptor work and rank operation syncs;
cache/fuse with complete invalidation/ownership and matched controls. All baselines/
experiments retained. Goal not complete. No install/push/supervisor/user-work changes.
Plan/work-log checkpoint separate from code/data; git diff --check passed.

Logs /tmp/xca-bs-{profile,arena,arena-profile}-{configure,build,run,stats,test,memcheck,campaign}.log
and /tmp/xca-bs-arena-large-smoke.csv. Persistent trace evidence under
experiments/block_sparse_cuda/scalar_paths/thinkstationP620_RTXA4000/profile-01,
paired results benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-arena-01.

## 2026-10-14 — Path-constant coefficient preparation

Previous goal turn only restated active status; this turn implements and qualifies
the next measured scalar optimization. 6068a8b [bench] broadcast scalar path
coefficients adds an isolated variant of the arena scheduler, preserving existing
variants. Shared recipe distinguishes node-independent coefficients from pole and
hybridization weights. Vertex and special endpoint coefficients, prefactor powers
and signs are computed once per lane on both CPU and GPU. GPU broadcasts a compact
lane vector using the existing explicit complex product formula, retaining operation
order and in-place semantics. No value caching, inverse, atomics, arithmetic
reassociation or production dispatch change. Temporary coefficients remain owned
through synchronization; the unchanged conservative workspace bound covers them.

Retained arena suite (13 tests) and constant variant (14 tests) pass both CTest
entries in 16.91 s. The extra kernel test compares broadcast products bit for bit
against expanded device weights for rank/lane shapes (3,17) and (23,512), changed
coefficients, zero lanes and in-place repeated multiplication. Six focused scalar
qualification/order/layout tests pass memcheck with full leak checking in 17.36 s:
zero errors, zero leaked bytes. CPU-only frontend builds; six-mode occupation SE,
order two, eight-index multi-cohort/tail run matches the original CPU exactly. Its
latency overlapped verification work and is not benchmark evidence. The new CPU
frontend explicitly requires bitwise control agreement. Existing production full
suite results remain applicable; this change only affects private benchmark fixtures.

All build/test/sanitizer handles were terminal before paired timing. 71bfca1 [bench]
record scalar coefficient timings preserves scalar-constants-01: 30 isolated
processes, 82 raw rows, hashes/commands, exact replay, and unchanged GPU discovery
counts relative to the arena campaign. All fifteen CPU cases have zero error against
original CPU. Maximum GPU screening relative error is 3.645e-13 in the same weak
higher-order non-scalar case; scalar ten-mode error is 4.926e-15. Campaign ten-mode
rows have one sample, other rows three. New GPU campaign ten-mode scalar latency
21.944 ms and matched CPU 20.744 ms; these are not merged with supplemental repeats.

After the campaign completed and retained CPU frontend rebuilt, a separate fresh
scalar comparison measured both old/new CPU and GPU variants, five timed samples
per case in one process after one warm-up. Ten-mode medians are CPU baseline 22.398
ms, CPU constants 20.993 ms, GPU arena 24.898 ms, GPU constants 21.179 ms. Four-mode
medians are 2.707, 2.687, 2.817, 2.636 ms respectively. Binary hashes, commands,
raw samples and min/median/max summaries are recorded in repeated-scalar. An initial
recorder syntax error occurred before execution; the corrected recorder produced
all eight successful cases / 40 samples. Prior controls and historical timings remain.

GPU improvement over fresh arena is about 14.9%, while matched CPU also improves
about 6.3%. Local CPU/GPU are at parity; this falls short of a robust promotion gate
and proves no GH200 dispatch, complete Solver/topology speedup or full N=14 workload.
GH200 remains the optimization target. Next separate residual synchronization/host
metadata costs and investigate multi-field special-vertex streaming (E6); current
singleton scalar batching does not optimize other shapes or correlators. Higher-order
cutoff-adjacent paths and complete host/opaque peak admission remain unqualified.
Goal remains active. No install, push, supervisor or unrelated user-work changes.
Plan/log checkpoint is separate from experiment/data commits; git diff --check passes.

Evidence: experiments/block_sparse_cuda/scalar_paths/README.md;
benchmarks/block_sparse_common/thinkstationP620_RTXA4000/scalar-constants-01/README.md.
Temporary logs: /tmp/xca-bs-constants-{configure,build,test,memcheck,campaign}.log,
/tmp/xca-bs-constants-cpu-{configure,build,qualification}.log or .csv,
/tmp/xca-bs-constants-{cpu-cuda-build,baseline-build}.log.

## 2026-10-14 — Bounded multi-field kappa retention

Previous goal turn made progress: 6068a8b/71bfca1/0fbf860 qualified scalar path
constants. This turn moves to E6 multi-field special vertices instead of extending
singleton-only microbenchmarks. Inspection confirms GPU reference recomputes every
kappa inside each mu loop, whereas original CPU retains all Tkaps. f3a66d8 [bench]
retain bounded kappa products adds private two-slot and sixteen-slot full evaluators
through a checked CMake source seam. Products use actual rectangular dimensions and
remain owned only while incoming T is unchanged. Mu/kappa traversal, reflected
coefficient indices, parity, weights and sequential additions are unchanged. No
sum factoring, CPU replay or production dispatch change. Global conservative extra
storage is capped by the largest nonsingleton set and admitted before input upload;
correlators inherit the extra allowance but do not allocate these cached products.
Singleton paths allocate no cache. Increasing the cap can reject previously fitting
workloads, so this is an experiment rather than a universal default.

Initial ten-test suites pass for both variants (15.11 s combined), including
bitwise comparisons against retained CUDA on four-field rectangular number-sector
paths, both sparse backends, complex G changes and zero masks. Extra payload math,
budget-minus-one rejection before path work, empty ranges and recovery pass. Two
focused tests per variant pass memcheck/full leaks: 8.00/6.99 s, zero errors/bytes
leaked. CPU-only configure passes and excludes these CUDA targets. Initial build
found const-view reassignment and generated-header include propagation mistakes;
both were corrected, regeneration/build/tests completed before timings.

After all verification handles were terminal, kappa-01 recorded 21 isolated
processes / 105 samples. Five timed calls per process after warm-up; not five
independent process repetitions. All passed CPU screening and exact device replay.
Discovery counts, warm norms and original-CPU control error metadata match the
ordered reference row for row. Raw GPU rows are in the CUDA benchmark hardware
folder; shared comparisons/manifests/hashes/analytic envelopes are in common.

Full-set medians versus GPU reference: N4 number NCA 11.506 vs 16.836 ms (31.7%),
N4 number OCA 8.854 vs 10.955 ms (19.2%), N6 number OCA 21.770 vs 30.828 ms (29.4%),
N8 number NCA 86.314 vs 136.789 ms (36.9%), N10 number selected NCA 296.941 vs
352.720 ms (15.8%). Two-slot retention gives smaller gains (3.5–15.1% on these
number cases). Mixed gains are modest; singleton occupation regresses about 1%.
N4/N8 NCA ranges include both directions; larger/higher-order calls select bounded
indices. These compare GPU routes, not hardware speedup against a matched CPU.
Original CPU control latency includes flattening and unchanged convolution; CPU
already caches kappas. No full Solver/topology-sum claim.

ce8c920 [bench] check bosonic kappa reuse adds a two-field shifted-density set
with complex off-diagonal dynamic coefficients, full NCA/reflection and changed G,
compared against retained GPU bit for bit. Final eleven-test suites pass twice:
ordinary CUDA 15.27 s combined, native-disabled library-only CUDA 15.10 s. Native
checks are conditional on actual capability; library controls run without native
kernels. Extra bosonic memcheck passes in 1.51 s with zero errors/leaks. The retained
library-only tree required Build_Benchmarks=ON for fixtures (previously OFF); both
XCA_CUDA_SMALL and XCA_CUDA_BLOCK_SPARSE remain OFF. This post-timing verification
did not overlap measurements. Final qualification/benchmark capability guard changes
leave the timed cache algorithm unchanged. Full production suites were not repeated
for isolated experiment changes; prior full production/MPI verification remains.

f19b5e1 [bench] record kappa retention timings preserves source/binary/header hashes,
commands, raw rows, medians and analytic memory envelopes. Binomial inventory avoids
N=14 atom_diag construction: N10/r23 extra two/full slots 0.044/0.218 GiB; N14/r26
extra 9.127/63.888 GiB beyond 90.409 GiB variable reference envelope. Fixed tables,
operator reserve and host/opaque storage are excluded from those analytic totals.
Actual cached rectangles may be smaller; no complete allocator peak proof. This
supports further streaming/map fusion rather than unlimited resident caching.

Next qualify direct sparse-column maps for kappa operators with at most one nonzero
per column, forming mu directly with original weighted-addition order and original
left application. Keep general CSR fallback, changed-input ownership, coefficient/
parity/reflection semantics and resource rejection explicit. Continue E7 memory and
GH200 target gates; no target/default promotion or N=14 production claim. All controls
and experiments retained. Goal active, not complete. No install/push/supervisor or
unrelated user-work changes. Plan/work-log commit remains separate; diff check passed.

Evidence: experiments/block_sparse_cuda/special_vertex/README.md;
benchmarks/block_sparse_common/thinkstationP620_RTXA4000/kappa-01/README.md.
Logs: /tmp/xca-bs-kappa-{configure,build,test,campaign}.log,
/tmp/xca-bs-kappa-{2,16}-memcheck.log,
/tmp/xca-bs-kappa-{boson,final,library}-{build,test}.log,
/tmp/xca-bs-kappa-boson-memcheck.log and cpu/library configure logs.

## 2026-10-14 — Mapped kappa weighting

Previous turn progressed through bounded kappa retention and recorded the memory
tradeoff. ad4393b [bench] fuse mapped kappa weighting implements the next E6/E7
candidate in a separate folder. A CPU-testable classifier validates CSR and builds
owned source/value tables when each column has at most one entry, preserving tiny
complex values and empty columns. Incompatible dimensions and multiple entries
reject the trait. The generated full evaluator keeps the original special vertex
as a reference fallback for non-mapping operators and singleton sets. A native
kernel directly forms mu in field order, with right product then weight then
sequential addition; the original left sparse application and mu-to-output addition
remain separate. Contraction disabled, no atomics/factoring/tree sum. Fusion can
change rounding versus library primitives; no universal bitwise claim or tolerance
changes. Shapes and checked map/weight allowance are guarded before input upload.
GPU maps/weights live through upload handoff and stream completion. No cross-call
numerical cache, public production change or altered Python/MPI interface.

The final suite has fifteen evaluator tests (eight unchanged physics/API plus
seven controls) and two host classifier cases. Both CUDA CTest entries pass in
7.49 s. Controls cover rectangular changed inputs/masks, complex off-diagonal
bosonic mixing/reflection, metadata-budget rejection/empty ranges/recovery,
non-mapping bosonic operators witnessed on accepted live paths, complex rectangular
kernel comparisons with changed weights/empty columns/tails, exact cancellation
field order, and nine representable amplitudes around a calibrated four-field NCA
completed-path cutoff. Kept and rejected cases both occur and contribution counts
match reference. Higher-order cutoff-adjacent behavior remains unqualified. The
fallback witness tightening independently passes its focused run. Seven focused
cases pass memcheck/full leaks in 9.65 s: zero errors, zero leaked bytes. Mixed
real/complex operator fixture addition needed explicit complex hopping operators;
corrected before final qualification. Two host cases pass CPU-only CTest in 0.02 s.

After all build/test/sanitizer handles were terminal, the shared recorder's new
--with-mapped option ran kappa-mapped-01: 28 isolated processes / 140 rows, five
samples each after warm-up, comparing reference, two-slot, sixteen-slot and mapper.
Every process passes screening and exact replay; discovery counts match all routes.
Number-symmetry reference/mapper medians: N4 NCA 16.904/4.519 ms (73.3%), N4 OCA
10.805/6.191 ms (42.7%), N6 OCA 30.544/12.673 ms (58.5%), N8 NCA 138.598/30.395 ms
(78.1%), selected N10 NCA 351.544/187.962 ms (46.5%). Mixed improves 18.3%; singleton
regresses 1.6%, within provisional protected screen. Maximum mapper screening error
8.567e-15. Five repeated calls in one process are not five independent process
repetitions. Original CPU control time is not an algorithm-matched throughput
benchmark; no CPU hardware speedup, full Solver/topology or target claim.

a4b8a14 [bench] record mapped kappa timings preserves raw GPU files, common samples/
medians, commands/source/binary/header hashes and analytic map allowances. Separately
collected post-timing build metadata records unchanged CMake flags/architecture and
nvcc version. N10/r23 map allowance 64,160 bytes; N14/r26 allowance 1,158,976 bytes
(1.105 MiB), versus prior full-set cache 63.888 GiB. This does not eliminate the
reference's 90.409 GiB variable envelope, nor fixed/operator/host/opaque storage.
No N=14 atom_diag or full evaluation was performed. All prior controls remain.

Post-campaign, native-disabled library-only configuration/build runs both host
classifier cases (CTest 0.13 s) and confirms the compiler-dependent mapper target
is absent. No overlap with timings. Both native production options remain OFF in
that tree; benchmark fixtures remain ON from prior verification. Full production
suites/MPI were not repeated for private fixture changes; prior integration remains.

Next qualify actual fourteen-mode Fock maps on bounded row slabs without atom_diag,
continue liveness/streaming admission, and qualify optional production/Python/MPI
integration of the promising route. Higher-order boundary and GH200 target gates
remain. Goal active, not complete. No install/push/supervisor or unrelated user-work
changes. Plan/log checkpoint separate from code/data; git diff --check passes.

Evidence: experiments/block_sparse_cuda/mapped_special/README.md;
benchmarks/block_sparse_common/thinkstationP620_RTXA4000/kappa-mapped-01/README.md.
Logs /tmp/xca-bs-mapped-{configure,build,test,campaign,memcheck}.log,
/tmp/xca-bs-mapped-{final,cpu,library}-{build,test}.log and configuration logs,
/tmp/xca-bs-mapped-fallback-{build,test}.log.

## 2026-10-14 — Actual fourteen-mode Fock row slices

Previous turn progressed through mapped-kappa full-call qualification. 1cc80db
[bench] replay Fock sector row slices adds canonical adjacent number-sector fields
without atom_diag/Hamiltonian construction, plus a bounded one-mu primitive. Both
annihilation and adjoint creation orientations use actual 3003/3432 shapes at fourteen
modes. Independent host tests verify known four-mode signs and fourteen-mode
completeness: 1716 entries per field, seven annihilation possibilities per upper
state and eight additions per lower state. Three host tests pass CPU-only (0.02 s)
and CUDA. Primitive rank/rows are bounded (26/128), rows must fit the source sector,
and explicit tensor/maps/control-operator envelope is checked against one GiB plus
free headroom before allocation. Fresh inclusive-route maps and duplicate resident
rank weights are accounted. Host/opaque allocator/context peaks are not claimed.

Four routes retain a matched serial CPU field-order sum, separate resident GPU
right/weight/add operations, resident mapped GPU, and fresh map rebuild/upload plus
mu download. Reference snapshots/rank weights are deliberately resident, favoring
that control over production snapshot rebuilding. CPU and maps/download return host
output within their timers; resident routes download outside timing. Input/output
GPU tensors remain allocated even for maps/download. Setup, warm controls and
validation remain outside timers. This is one mu slice, not a complete special
vertex/path/solver. No left operation, sum over mu, cutoff, DLR, full output or MPI.

Fourteen-mode/rank3/17-row smoke passes. Inverse creation slice passes final
memcheck/full leaks with zero errors and zero bytes leaked. Sanitizer timings are
excluded. All verification handles were terminal before isolated measurement.
d45c698 [bench] record Fock slice timings retains eight processes / 96 rows:
both directions for N4/r3/rows3, N10/r26/rows32, N14/r26/rows32 and rows128.
Three timed samples per route are one process, not independent repetitions.
Source/executable hashes, toolchain/hardware and commands are recorded. All strict
1e-13 relative controls and exact mapped replay pass; mapped CPU/device errors are
zero for this finite suite. Serial/compiler CPU control is not a tuned/threaded
CPU ceiling; small four-mode CPU remains faster than GPU.

N14/r26/rows128 medians: CPU 682.044/637.464 ms, separate resident GPU 43.813/38.943
ms, mapped resident 20.281/17.784 ms, fresh maps+download 111.836/96.341 ms
(annihilation/creation). At rows32, mapped resident 5.123/4.497 ms, maps+download
29.299/24.698 ms. Materialization can dominate: keep mu on-device through consumers.
These primitive results do not qualify whole-solver throughput, complete live peak,
N=14 full-call admission or GH200 performance. No full N=14 model was constructed.

Next integrate an explicit production mapped option with unchanged reference
defaults and full CPU/CUDA/library-only/Python/archive/MPI qualification, then
continue E7 scheduling and higher-order boundary gates. GH200 remains the target.
All baselines/experiments retained. Goal active, not complete. No install/push,
supervisor or unrelated user-work changes. Plan/log checkpoint separate from code/
data; git diff --check passed.

Evidence: experiments/block_sparse_cuda/mapped_special/README.md and its
thinkstationP620_RTXA4000/fock-slabs-01/README.md. Temporary logs:
/tmp/xca-bs-slab-{configure,build,final-build,campaign}.log,
/tmp/xca-bs-slab-{cpu,host}-test.log, /tmp/xca-bs-slab-final-memcheck.log,
/tmp/xca-bs-slab-smoke.csv; CPU configure/build logs.

## 2026-10-14 — Production mapped policy and MPI integration

Previous turn made progress through actual fourteen-mode primitive slices. f58aad5
[bs] map CUDA special vertices promotes the qualified column classifier/kernel and
special vertex into the production CUDA evaluator, with optional SpecialBackend::Mapped
and default Reference. SparseBackend remains independent. XCA_CUDA_BLOCK_SPARSE
builds both native kernels; compiler-disabled CUDA rejects explicit mapped requests.
Constructor validates policy/capability before creating operator/DLR work. The Python
facade validates capability before model export/device setup. Maps and weights own
their storage through stream completion and default-upload handoff. CSR eligibility,
shape checks, field order, original left application and sequential mu addition are
retained. Unsupported columns/singleton sets choose the device reference before any
mapped work; failures propagate without CPU replay. Reported/admitted explicit
storage adds checked metadata allowance only for mapped policy. Correlator admission
conservatively includes it although correlators retain original execution.

Production tests port seven meaningful experimental controls to the real evaluator
and add invalid-policy/unavailable-capability checks. Direct primitive calls use the
production native kernel and classifier. Eight controls pass in the native build;
compiler-disabled builds omit native primitive tests and check rejection. CPU has
the expected disabled CUDA case, not GPU validation. Initial full build exposed
duplicated policy enum/function definitions in generated private copies. The enum
moved to shared DeviceOps declarations and capability definition to its shared
translation unit. All retained experiments then build without altering their default
reference policy, numerical controls or ownership. No production tolerance changes.

aa014e5 [solv] select mapped CUDA special vertices adds special_backend='mapped' to
both real/complex consolidated facades and block_sparse_special_backend to Solver.
Reference remains default. gpu.py validates/lazily loads and checks compiled support
before device/model setup; CPU stores preference unused. TOML explicitly exports
block_sparse_mapped_available; only consolidated CUDA bindings were regenerated
with clair and optional-small guards restored. Archive/deep-copy/old-archive defaults
are tested. Public documentation describes eligibility, budgets, independent sparse
selection and target limits. Python registration tests cross both special policies
with available library/ordered backends, all three partitions, dynamic variants and
SE/SPGF/correlators. Collective initialization/count/compute failures and retry also
run with mapped policy, plus bad selector and compiler-disabled rejection.

All full builds and suites pass: CPU 79/79 in 50.39 s; CUDA 96/96 in 673.45 s;
library-only CUDA 81/81 in 353.91 s. CUDA trees initially competed for the same GPU,
substantially increasing wall times. A stop was prepared for the overlapping library
CTest, but authoritative inspection found it had already completed successfully,
so no signal was sent and no suite restarted. These durations are correctness
verification, not benchmark data. Compiler-disabled tree keeps both native options
OFF and benchmark fixtures ON. Production memcheck, after full GPU suites were
terminal, passes all eight controls in 9.45 s: zero errors, zero leaked bytes. The
first sanitizer invocation used a wrong executable path and performed no work;
corrected path came from CTest's registered command.

Two-rank MPI then ran the actual five-method Python suite using CTest environment
modifications (build package plus cached adapol/pyed), OMP/BLAS threads one. Both
ranks pass all five methods in 92.742 s, including mapped/archive/collective recovery
checks. Harmless existing X11 authorization diagnostics appear before execution;
there was no tool approval rejection. No production timing claim from MPI/tests.

Goal remains active. Next refine schedule-sensitive E7 admission and scratch
lifetimes; distinguish NCA from convolution-bearing paths and retain complete API
output accounting. Higher-order cutoff-adjacent coverage, broader batching and GH200/
full N=14 qualification remain. No new target/default crossover is claimed. All
baselines/experiments retained. No install, push, supervisor or unrelated user-work
changes. Code and integration commits are separate from this plan/log checkpoint;
git diff --check passes.

Evidence: doc/python_api.rst; test/c++/block_sparse/cuda_mapped_special.cpp;
test/python/block_sparse_cuda.py. Temporary logs:
/tmp/xca-bs-mapped-prod-{cpu,cuda,library}-test.log and associated build/configure
logs; /tmp/xca-bs-mapped-prod-{memcheck,mpi}.log;
/tmp/xca-bs-mapped-api-{configure,wrap,build}.log.

## 2026-10-14 — Compact DLR liveness experiment

Previous turn progressed through verified production mapped integration. Inspection
of actual DlrOps ownership found cached solve RHS and five retained convolution
buffers. 13ab3b0 [bench] compact DLR scratch adds an isolated source snapshot and
full evaluator control, preserving production defaults/ABI and all earlier routes.
Completed RHS is released at convolution entry. Disjoint output stores the diagonal
term, saving its buffer/final copy; overlapping inputs retain the owned diagonal.
Overlap/shape/budget checks precede release/output writes. Same LU, products and
addition order, no inverse/factoring. Actual rectangles and alias-mode cache keys
are retained. Full evaluator reserves five DLR tiles because its outputs are
separate; generic alias cases may require six after a later solve.

Four primitive controls pass (0.63 s), including original real-mesh/rectangle/LU,
reflection, cancellation, rejection/recovery and independent exact production
comparison/storage/alias switching. Memcheck/full leaks passes all four in 0.64 s:
zero errors, zero leaked bytes. Eight inherited complete observable/API controls
pass in 7.30 s with the compact reference policy. These are private fixture changes;
no full production/Python/MPI suites repeated or installation performed.

After verification handles were terminal, compact-01 runs six isolated composed
edge cases / 36 rows with two solves, packing/status waits, convolution and allocation/
release inside timing. Tables/input uploads/caller arrays/download checks outside.
All warm/replayed outputs agree exactly with cached production in this finite suite.
At rank18/D256, cached 51.090 ms vs compact 51.346 ms. Cached retained/boundary bytes
113,272,204; compact retained 75,523,468 (36 MiB saved), boundary peak 94,397,836
(18 MiB saved). The next RHS coexists with four scratch tensors. This is a method-
boundary explicit observation, not a continuous allocator/opaque/host/caller peak.
Small/rectangular case timing changes are also modest; no robust speedup claimed.

The five-route complete-call screen compact-full-01 records 35 processes / 175 rows,
five repetitions per process, covering ordered reference, two caches, mapped and
mapped/compact. All pass CPU screening and exact replay; discovery counts, warm
norms and error metadata match mapped/compact row for row. Changes range -3.9%
to +1.4%, within local noise/protected-case screen. The N10 declared bound decreases
473,446,400 to 450,076,928 bytes. NCA does no DLR work, so this is conservative reserve
refinement, not measured released NCA storage. N14/r26 extrapolated reserve saves
4.563 GiB; no full fourteen-mode model/evaluation or fitting proof. No tuned/threaded
CPU or whole Solver/target speedup claim.

b5aad7e [bench] record compact DLR storage preserves primitive/full raw samples,
commands/source/executable hashes, comparisons and scope. These runs are isolated
from tests/profilers. Next qualify schedule-specific NCA/convolution admission and
scratch lifetimes before promoting a memory policy. Retain full output, host/opaque
and higher-order boundary gates, plus GH200 validation. Goal active, not complete.
No install/push/supervisor or unrelated user-work changes. Plan/log commit separate
from code/data; git diff --check passes.

Evidence: experiments/block_sparse_cuda/compact_dlr/README.md and its hardware
compact-01/README.md; benchmarks/block_sparse_common/thinkstationP620_RTXA4000/
compact-full-01/README.md. Logs /tmp/xca-bs-compact-{configure,build,test,memcheck,
campaign}.log; /tmp/xca-bs-compact-eval-{configure,build,test}.log;
/tmp/xca-bs-compact-full-campaign.log.

## 2026-10-14 — NCA-specific admission and scratch release

Previous turn progressed through compact DLR liveness. e051a83 [bench] qualify NCA
storage admission adds a private full evaluator/snapshot, preserving production
APIs/defaults and all previous experiments. The topology-specific self-energy bound
validates topology and removes six inactive DLR scratch tiles only for one-line NCA.
Generic no-argument estimate, higher orders and correlators remain unchanged. Full
functions, seven caller tiles, fixed tables, sparse reserve, kernel weights and mapped
metadata are still counted. Budget rejection happens before state changes. After
that check, eligible NCA releases completed DLR RHS/convolution scratch before the
free-memory/headroom query and before input/output upload or path arithmetic.

Private release requires construction device and stream completion, destroys scratch
and resets shape cache while retaining immutable rank/LU tables, solver and shared
handles. It is idempotent; subsequent solves/convolutions reconstruct scratch. This
addresses OCA/NCA/OCA reuse, so old retained allocation cannot sit outside the lower
reservation. No CPU numerical replay, arithmetic changes or fallback after failure.

Ten tests pass in 8.01 s: eight unchanged observable/API tests plus lower-budget
acceptance, bound-minus-one rejection with zero path/peak counters, empty ranges,
higher-order rejection/recovery, exact production comparisons and inter-order
transitions for reference/mapped special policies. Independent release/solve/
convolution reconstruction is exact and storage returns to table-only bytes. Two
focused cases pass memcheck/full leaks in 7.40 s: zero errors, zero leaked bytes.
No new performance benchmark needed for pure reservation proof; these timings are
correctness evidence. No full production/Python/MPI suites repeated for private
fixture changes, and no wrapper regeneration or installation.

Analytic binomial N10/r23 and N14/r26 envelopes are preserved in the experiment's
hardware folder, without costly atom_diag construction. N14 variable reserve falls
90.409 to 63.029 GiB, omitting six scratch tiles only. Complete input/output remains;
fixed/operator/kernel/mapped/host/opaque storage is additional. This is not a full
fourteen-mode admission/peak or GH200 claim. CPU-only configuration remains guarded;
native kernels are needed only for mapped special policy, not reference admission.

Next promote the bound overload and release into production with controls against
topology-specific reservation; update private scalar scheduler seams to retain their
extra workspaces and consolidated Python/MPI exposure/qualification. Keep generic
upper accessor and defaults. Higher-order boundaries, wider streaming/host/opaque
peaks and target verification remain. Goal active, not complete. No install/push,
supervisor or unrelated user-work changes. Plan/log commit separate from code/data;
git diff --check passes.

Evidence: experiments/block_sparse_cuda/nca_admission/README.md and
thinkstationP620_RTXA4000/{README.md,analytic_envelopes.csv}. Logs
/tmp/xca-bs-nca-{configure,build,test,memcheck}.log.

## 2026-10-14 — Production NCA admission

Previous turn qualified private NCA reservations. 30685cf [bs] scope NCA storage
admission promotes the topology-aware estimate and DlrOps release into production.
Generic no-argument upper estimate remains. Validated one-line SE calls subtract
the configured DLR scratch reserve, keeping tables and all input/output/caller/
operator/weight/mapped metadata reserves. Explicit-budget rejection precedes state
changes; eligible NCA releases completed retained scratch before free-memory query
and input upload. Empty ranges remain valid. Release synchronizes only when storage
exists, resets RHS/convolution shapes and preserves tables, solver/shared handles.
Arithmetic and default backend policies are unchanged.

Compatibility work keeps all experiments: scalar generated admission and topology
accessors add scalar_extra_bound; compact DLR implements release; private NCA now
uses the production admission seam. Budget-minus-one tests in mapped/scalar/kappa
controls use the actual topology estimate. Production C++ controls independently
check reference/mapped tight acceptance, rejection/no-work, empty/higher-order
recovery, OCA/NCA/OCA transitions, exact comparisons and table-only release/rebuild.
1c18718 [solv] expose topology storage bounds adds the consolidated overload,
regenerated/guarded only CUDA wrappers, public documentation and Python/MPI tests
for estimates, lower-budget acceptance, OCA rejection and reusable later calls.

Two CUDA builds were accidentally started concurrently while final tests were being
added. Only their identified cmake/descendant processes were explicitly stopped;
both handles terminated. One final serial build then completed. No measurements
or checks from discarded builds count as verification. CPU and library-only builds
use separate trees; GPU suites ran sequentially between trees.

CPU full suite 80/80 passes in 50.36 s. Main CUDA full run: 98/100 passed in 518.35 s;
kappa_2 and kappa_16 still had stale generic-bound-minus-one rejection assumptions.
The new NCA call correctly fit that budget. After changing both controls to the
actual topology estimate, focused two-entry rerun passes in 15.33 s. The other 98
entries remained valid (only retained test fixture changed); all 100 final covered
entries pass. Full native-disabled library-only suite 85/85 passes in 249.05 s.
Both native options stay OFF in that tree. These are correctness wall times, not
performance data; small GPU tests contend within CTest parallelism.

After suites were terminal, production admission memcheck/full leaks passes both
controls in 7.81 s: zero errors/bytes leaked. Two-rank Python then passes six methods
per rank in 99.341/99.345 s using CTest build-tree/dependency environment and OMP/
BLAS threads one. Coverage includes actual topology-bound calls, budget failure
collectives and recovery across NCA/OCA for both special policies. Existing harmless
X11 authorization diagnostics precede MPI; no tool approval rejection occurred.

Goal active, not complete. NCA bound is now public, without changing arithmetic or
claiming a whole-solver speedup. Full output, broader batching/compact scheduling,
higher-order cutoff boundaries, host/opaque peak and GH200/full N=14 qualification
remain. Next audit supported-call E4/E7 selection and conditional E8 evidence. All
baselines/experiments retained. No install/push/supervisor or unrelated user-work
changes. Code, integration and plan/log commits remain separate; diff check passes.

Evidence: doc/python_api.rst; test/c++/block_sparse/cuda_admission.cpp;
test/python/block_sparse_cuda.py. Logs /tmp/xca-bs-nca-prod-{configure,wrap,build,
cpu-test,cuda-test,library-test,memcheck,mpi}.log and associated build/configure logs;
/tmp/xca-bs-nca-{kappa,library-kappa}-build.log; /tmp/xca-bs-nca-kappa-test.log.

## 2026-10-14 — Bounded topology prefix discovery

Added the CPU-only topology_reuse experiment and benchmark CMake target. It uses
the actual diag.py generator, scans at most 4096 pairings, selects at most six
connected topologies, then visits at most 4096 surviving paths for two flat indices.
Keys conservatively encode ordered pre-special operations and fixed input identity.
Special sums, suffixes, final parity/cutoff and topology reduction are excluded;
no numerical sharing or production policy change is implemented.

The prefix-01 workstation campaign completes 24 cases / 72 stage rows for four
modes, number/mixed/occupation partitions, orders 2/3/4/6 and active/periodic-mask
inputs. Selection scans 3/15/38/1088 pairings, retaining 1/4/6/6 topologies. Normal
cases are not truncated; separate actual executable controls check path-limit
truncation and malformed input rejection. Empty paths produce explicit zero rows.
Python compilation, CPU build and recorder checks pass. CUDA build and an order-4
number probe match the CPU CSV exactly. This is host discovery, not GPU testing
or timing, so existing full-suite correctness results remain the prior checkpoint.

Number symmetry produces 8/8/4 shared edge keys at orders 3/4/6, with 20/24/20
distinct-topology reuses. Selected mixed order-6 paths and all masked cohorts have
no cross-topology overlap. Root-copy reuse is counted separately. Discovery counts
support a bounded numerical replay experiment, not a cache or graph promotion.
Literal input generation markers apply only to this immutable fixture; a future
cache must handle all input changes and owned lifetimes. Initial selection bias,
full-call setup/memory costs, cutoff behavior and target hardware remain open.

Next qualify shared prefix values independently, then compare bounded numerical
DAG execution against independent calls before attempting graph capture. Existing
public allocation/synchronization and host cutoff paths are not capture-ready.
Goal remains active. Target GH200-class hardware, not the RTX A4000. All experiments
and user work preserved; benchmark changes and plan/log updates committed separately.
No installation, push or supervisor changes. Logs /tmp/xca-bs-reuse-{configure,build}.log
and /tmp/xca-bs-reuse-cuda-{configure,build}.log; retained machine manifest and CSVs
under experiments/block_sparse_cuda/topology_reuse/thinkstationP620_RTXA4000/prefix-01.

## 2026-10-14 — Numerical topology prefix qualification

Previous goal turn classified as progress: committed bounded structural discovery
and actual CPU/CUDA-build evidence. This turn adds optional independent numerical
replay to the retained topology_reuse experiment, without production arithmetic or
default changes. Each path recomputes CSR/hoisted-CPU prefixes and a separately
advanced dense-operator/original-CPPDLR reference. All repeated structural keys
must produce exactly equal tensors. Finite checks precede comparisons; stored
candidate tensors are capped at 64 MiB across stages. The reference screen remains
1e-10 times signal plus 1e-14, not a new universal solver accuracy promise.

Final replay-01 campaign completes 24 bounded cases / 72 rows and 384 exact repeated
value checks. Independent-reference max absolute error is 8.423845684525094e-18;
largest all-stage candidate payload is 204,608 bytes. Changed propagators (1.01x)
and coefficients (.97x) in fresh processes yield changed non-root checksums while
within-call repeated keys remain exactly equal. These controls prove recomputation
needs, not implemented cross-call invalidation. Path-cap truncation, malformed
input and low-overlap controls pass. Initial uncommitted run was replaced after
finite guards/rebuild so the final manifest matches the executable/source hashes.

CPU target build and complete recorder pass. CUDA target build and order-4 number
numerical replay reproduce CPU CSV byte-for-byte. These host operations do not
constitute GPU numerical or full-suite regression testing. Logs:
/tmp/xca-bs-reuse-replay-build.log, /tmp/xca-bs-reuse-replay-cuda-build.log,
/tmp/xca-bs-reuse-replay-cuda.csv. Final data/manifests are retained under the
experiment's thinkstationP620_RTXA4000/replay-01 folder.

Goal remains active. Next test a bounded call-local device cache against independent
complete evaluations, including discovery/misses/storage and unchanged special,
cutoff and reduction order. Selection bias, broader input-update behavior and
GH200-class measurements remain unresolved. No speedup, graph capture or full-system
N=14 qualification is claimed. Code/data and plan/log commits stay separate; no
push/install/supervisor or unrelated work changes.

## 2026-10-14 — Bounded device prefix cache

Previous turn was progress: independent CPU prefix qualification and hashes/data
were committed. Added a generated private PrefixEvaluator under topology_reuse,
retaining production arithmetic except reusing exact post-edge intermediates.
Model/poles/mesh/coefficients are immutable per instance. A complete byte-for-byte
G snapshot invalidates entries before a changed-input call, including zero masks.
Host snapshot cap is 32 MiB (old/new peak up to 64 MiB); larger inputs clear/disable
cache while retaining independent device execution. Stored tensors plus conservative
metadata charges are capped at 8 MiB/4096 entries; misses beyond it recompute.
Admission charges the cap, subtracts only actual already-resident device payload
and includes retained cache in correlator bounds. Clear synchronizes the owner.
Special sums, suffixes, per-path cutoff and output/topology reduction remain ordered.

Native-enabled inherited eight evaluator controls pass (one CTest, 6.86 s), covering
self-energy, SPGF, bosonic/fermionic/dynamic correlators, masks, cutoffs, range/error
recovery and long-double reference comparisons. CPU-only configure passes. The
native-disabled tree keeps both native options OFF and passes the same eight private
evaluator controls (one CTest, 6.78 s). Library and Ordered/Mapped memcheck/full leaks run the complete
order-3 number cold/replay/changed-G benchmark: zero errors and zero bytes leaked.
No profiler/test/build work overlapped final timing samples.

Final cache-01 contains 18 processes / 162 paired samples over 3 orders and 3
partitions, each Library/Reference and Ordered/Mapped, three cold/replay/changed-G
samples. Every per-topology output matches the independent device evaluator exactly.
The first uncommitted campaign was replaced: G changed only on the first sample
and Library alone was measured. Final G changes before every changed-G sample;
Ordered/Mapped is included. Cold cleanup outside timing is explicitly excluded;
input comparison, misses, discovery, population, copies and transfers are included,
and changed-G invalidation/cleanup is timed. Construction/Python/MPI are excluded.

Ordered/Mapped order-3 number independent/cached medians: cold 62.884/49.439 ms,
replay 63.214/39.347 ms, changed G 63.308/49.422 ms. Changes reduce time 21.4/37.8/
21.9%. Order-4/6 number changed G reduces 13.2/7.1%. Order-3 mixed reduces 18.0%,
but order-6 mixed increases .6% without cross-topology prefix overlap. The selected
occupation order-4/6 paths do no prefix work; small overhead there cannot characterize
the whole workload. Largest charged cache is 96,800 bytes; no cap saturation claim.

Goal stays active. Selected cohorts justify further graph-free sharing experiments,
not unconditional production promotion. Next force saturation and tails, broaden
index/topology selections, and evaluate setup/selection overhead. GH200-class target
qualification and broader memory/batching/cutoff gates remain. All baseline and prior
experiments retained; code/data and plan/log commits separate. No push/install or
supervisor changes; unrelated user work preserved. Evidence is retained cache-01
manifest/samples and /tmp/xca-bs-prefix-cache-{configure,build,test,memcheck}.log,
/tmp/xca-bs-prefix-cache-final-{build,memcheck,campaign}.log and associated CPU/library
configure/build/test logs. Full-suite verification remains the earlier production
checkpoint; these are isolated experiment controls.

## 2026-10-14 — Forced prefix-cache limits and tails

Previous goal turn was progress: private device caching, paired complete-call data,
full observable controls and sanitizer evidence were committed. Added private lower
byte/entry/snapshot limits with validation before state changes. Default maxima
remain 8 MiB / 4096 entries / 32 MiB snapshot; admission uses the configured byte
limit. Added observable entry/skipped-store/snapshot-capacity diagnostics for tests.
Clear now releases host vector capacity, avoiding retained snapshot storage after
reconfiguration or bypass. No public evaluator, wrappers or backend defaults change.

Two new qualification controls use Library/Reference and Ordered/Mapped. For each
number/mixed/occupation fixture they force one-byte rejection, one-entry saturation,
one-byte snapshot bypass and normal caching. Independent complete outputs match
exactly for uneven ranges [0,1), [1,3), [3,4), repeated traversal, changed G, periodic
zero masks and restored G. Actual one-entry number controls require both cache hits
and skipped stores. Invalid oversized configuration and empty calls preserve state;
clear leaves zero entries and zero snapshot capacity. Every actual output is finite.
This widens coverage to four selected indices but is still not a population sample.

Initial Library control plus eight inherited tests passed in 14.31 s. Extending
capacity assertions initially conflated byte-cache rejection with host-snapshot
bypass; corrected the assertion to require zero snapshot only for bypass/clear.
Final native-enabled one-entry CTest passes all ten controls in 17.40 s. Native-
disabled build retains both native options OFF: nine pass, Ordered/Mapped is explicitly
skipped, CTest passes in 14.25 s. GPU checks ran sequentially between trees. Optimized
forced-limit memcheck/full leaks passes the full new control in 22.228 s, zero errors
and zero bytes leaked. Earlier cache-01 timings remain historical evidence for their
hashed implementation; no new timing or unchanged performance claim is made here.

Goal active. Forced storage limits and selected range tails are now qualified;
broader topology/index choices, cache-selection/setup policies, remaining memory/
batching/cutoff gates and GH200-class qualification remain. Experiments/baselines
and unrelated user work preserved. Code and plan/log commits separate, no push,
installation or supervisor changes. Logs /tmp/xca-bs-prefix-limits-{configure,build,
test}.log, /tmp/xca-bs-prefix-limits-final-{configure,build,test}.log,
/tmp/xca-bs-prefix-limits-library-{configure,build,test}.log and
/tmp/xca-bs-prefix-limits-memcheck.log. Full production suites remain the prior
checkpoint; these focused checks qualify the private cache experiment only.

## 2026-10-14 — Distributed topology and index selections

Previous goal turn was progress: forced cache limits, capacity release, tails and
both native configurations were verified and committed. Extended the private cache
benchmark with bounded first/count/modes selection, validating every topology domain
before timing. Original CLI remains supported. Paired complete-call comparisons now
also require exact accepted and contributing path counts and finite cached outputs.
No production evaluator or integration policy changes.

New selection-01 recorder scans at most 4096 actual diag.py pairings, then spreads up
to six connected topologies across the bounded set. Order-3/4 scans exhaust 15/105
pairings and select 4/6 of 4/27 connected topologies. Order 6 scans only 4096 of
10,395 pairings, selects six of 963 connected members; this remains biased rather
than population sampling. First indices 0,7,31 cover three indices in each four-mode
number/mixed/occupation case. One extra six-mode number order-3 case uses first 7.
Discovery time and actual selections are retained; discovery/model setup are outside
device timing. Snapshot invalidation, key generation, misses/copies and transfers
remain inside changed-G complete-call timings. No concurrent builds/tests/profilers
overlapped the timing campaign.

28 processes / 252 samples pass exact output and accepted/contributing count checks.
Largest charged cache is 2,278,656 bytes; forced saturation evidence remains the
earlier limit tests. Active changed-G order-3 four-mode gains are 18.0–27.1%; six-mode
number medians are 202.672/159.149 ms independent/cached (21.5% reduction). Active
order-4 gains are 11.1–20.3%, order-6 number 8.9–9.0%; mixed order-6 ranges from .7%
regression to 1.0% reduction with zero/two hits. Occupation order-4 first 0 and all
selected order-6 have no accepted paths, excluded from active-work gain claims.
Accepted-but-output-cutoff-rejected number paths agree on both routes; this is not
a separately constructed cutoff-adjacency proof.

Benchmark build, final source-hash check and invalid first/count/modes argument
controls pass. After timings terminate, distributed mixed order-4 first31/count3
memcheck/full leaks passes all nine complete-call samples with zero errors/leaks.
These are benchmark qualification checks; earlier full production suites and private
ten-control regression remain the prior checkpoints. No new target-hardware timing
claim. Data/manifest under topology_reuse/thinkstationP620_RTXA4000/selection-01;
logs /tmp/xca-bs-prefix-selection-{build,campaign,memcheck}.log.

Goal stays active. Broader evidence supports selective graph-free sharing but not
universal promotion. Next measure reuse-selection overhead versus always-on/off
execution and wider/random index cohorts; audit remaining full-function memory and
many-small batching gates. GH200-class hardware remains the target, with local
RTX A4000 results only a development proxy. All baselines/experiments and unrelated
user work preserved. Code/data and plan/log commits separate; no push/install or
supervisor changes.

## 2026-10-14 — Production compact DLR checkpoint

Previous goal turn was progress: broader cached-topology selections and paired exact
outputs/counts were committed. This turn returns to the large-block memory path and
promotes the qualified compact DLR storage policy. It releases completed solve RHS
before convolution, uses disjoint output for the diagonal term and retains a separate
diagonal for input/output aliases. Arithmetic order and public signatures remain
unchanged. Cached square scratch is four tiles; next solve is five. Evaluator reserve
is five maximum-size tiles instead of six. NCA still omits all inactive scratch;
its generic-minus-NCA test now expects five, and disjoint budget control uses the
smaller required allocation. Added production storage/alias/release transition test.

Preserved original DLR as compact_dlr/historical_dlr.{hpp,cpp}; primitive independent
cached/compact comparisons explicitly use that baseline. Private compact generated
evaluation accepts the new five-tile reserve. Private NcaDlr retains original cached
storage; its generated higher-order reserve stays six to preserve prior controls.
All experiments/baselines remain available. No new timing claim; prior campaign
hashes continue to describe their earlier versions. Existing N14/rank26 analytic
variable generic envelope becomes 85.845 GiB from 90.409, saving 4.563 GiB per maximum
tile. This excludes fixed/operator/host/opaque storage and is not actual N14 runtime.

Initial main/native-disabled builds failed because the retained baseline header
used a private include; corrected it to the public project device_ops path and
restarted each only after its actual build handle terminated. Final main and
native-disabled full builds pass. CPU configure/build/full CTest passes 80/80 in
50.51 s. Main focused six CTests pass in 25.10 s, including production DLR four
controls, admission, all-observable evaluator and private compact/NCA controls.
Production test listing confirms CompactStorageAndAliasTransitions is in the binary.

Verification remains in progress: unified exec session 23832 runs full main CUDA
CTest (log /tmp/xca-bs-compact-prod-cuda-test.log), then production alias/storage
memcheck (log /tmp/xca-bs-compact-prod-memcheck.log). Re-poll that exact handle; do not
restart because observation times out. After it is terminal, run the native-disabled
full suite sequentially on the GPU; its build session 54715 is already terminal
and successful. CPU session 8373 and main focused session 13293 are also terminal.
Do not claim full GPU verification from the focused controls. No other GPU test
suite or timing campaign is running. Follow up on failures before declaring this
production change fully qualified. Logs /tmp/xca-bs-compact-prod-{configure,build,
build-final,focused}.log and associated cpu/library configure/build/test logs.

Goal active. Keep full-function capacity, broader batching/cutoff and target-hardware
gates open; GH200-class remains the target and the RTX A4000 a development proxy.
Code verified by focused controls/CPU suite is checkpointed separately from these
plan/work-log updates. No push/install/supervisor changes; unrelated work preserved.

## 2026-10-14 — Compact regression and complete-output streaming

Previous goal turn was progress: production compact DLR was committed with focused
and CPU verification; full GPU checks remained live. Re-polled exact session 23832
without restarting. Full main CUDA suite passes 101/101 in 565.48 s; queued production
CompactStorageAndAliasTransitions memcheck passes, zero errors/bytes leaked. Native-
disabled full suite then passes 86/86 in 269.69 s with both native flags OFF. These
complete the broader compact regression gates; full CPU remains 80/80 in 50.51 s.

Prepared an isolated OutputSectorEvaluator generated from production, preserving
all observables and the existing SE visitor/arithmetic/cutoff. SE inputs remain fully
resident, but only one output sector resides on GPU. Sector switches finish the owner,
download the completed sector, release its device storage and resume the next sector
from host values. All additions stay on GPU in original order. Complete host output
includes zero/unvisited sectors. Return moves the vector into block_gf; the installed
TRIQS owning constructor takes/moves its vector, avoiding a second whole-output copy.

Self-energy device bound substitutes one maximum tile for one full function, retaining
existing correlator/SPGF implementation and bounds. Private host output payload budget
(default 512 MiB) is checked before allocations/uploads even for empty ranges. It
excludes caller G, GF metadata/allocator overhead and a transient maximum-sector
download; no RSS cap claim. Errors propagate, with no CPU summation or evaluator replay.
Prior N14/rank26 analytic variable envelope is 74.867 GiB versus compact resident
85.845 GiB, saving 10.979 GiB. This excludes fixed/operator/host/opaque memory and is
not actual N14 atom_diag or production qualification.

Generator seams and CPU-only configure pass. Main private target's nine controls
pass in 8.01 s; native-disabled nine pass in 8.27 s. Eight inherited tests cover all
observables/API behavior; added complete-output control checks three partitions,
exact outputs/counts, changed G/masks, empty output, host budget rejection/recovery
and a tight device budget: resident evaluation rejects, streamed complete output
matches the independent resident evaluator exactly. Focused complete-output memcheck
passes in 7.263 s with zero errors/leaks. All GPU stages run sequentially. No timing
or measured peak claim is made; full 101/86 suites preceded addition of this new CTest,
which is qualified separately rather than falsely reported as full 102/87 suites.

Session 60685 completes the production two-rank Python integration after the
native-disabled streaming check: six methods per rank pass in 98.016/98.020 s.
The CTest environment is derived from JSON properties, with OMP/BLAS threads one.
Log /tmp/xca-bs-compact-prod-mpi.log includes the known X11 startup diagnostics;
these are not tool approval rejection. All verification handles are terminal at
this checkpoint; no build/test/profiler remains running.

Next measure full-call sector transfer/setup overhead and explicit/measured peak
storage before deciding on promotion; input residency and spatial/K slabs remain
separate capacity gates. Goal active, GH200-class target, local RTX A4000 only a
development proxy. All baselines/experiments/user work retained; code/plan/log commits
separate; no install/push/supervisor changes. Logs /tmp/xca-bs-compact-prod-{cuda-test,
memcheck,library-test}.log and /tmp/xca-bs-output-sector-{configure,build,test,memcheck,
cpu-configure,library-configure,library-build,library-test}.log.

## 2026-10-14 — Complete output-streaming measurements

Previous turn was progress: full compact verification and nine-control output-sector
experiment were committed. Added a complete-call benchmark with physical fixtures,
topology-specific bounds, setup/call times, exact outputs/counts, signal, sampled
device deltas and process RSS. Initial outputs-01 two-route comparison has 98 rows.
Recognized that streamed return explicitly moves its vector while resident return
does not; added a generated resident explicit-move control before drawing conclusions.
Final outputs-02 has seven processes / 147 rows, three routes and first/warm/per-
sample changed-G phases. Route order rotates over three samples. First rows share
context/JIT state; setup is separate, transfers/allocations/comparison-independent
calls are timed, no Solver/MPI gain claim. Static number/mixed/occupation fixtures
cover orders 1–3, offsets and 4/6/8/10 actual modes, not N14 atom_diag.

All outputs and accepted/contributing counts match exactly. Bounds differ by full
output payload minus maximum tile, verified for every pair; ownership-only control
has the resident bound. Device counter is a selected-point global free-memory drop
from call entry, includes allocator/other-process effects and can be zero on small
cases. RSS is the cumulative paired/three-route process peak, not per-route host
storage. No continuous/private peak or PCIe bandwidth claim. Timings ran isolated
without builds/tests/profilers. Native-disabled benchmark builds and all nine small
control rows match using Library/Reference. Focused three-route optimized memcheck
reports zero errors/leaks; its handle terminated before library runtime control.

Ten-mode changed-G medians are 204.181/162.227/171.698 ms resident/move/streamed;
streaming costs 5.8% relative to move. Eight-mode values 21.355/18.835/18.891 ms cost
.3%; smaller changed-G cases cost .2–1.2%. Ten-mode sampled changed-G deltas are
234,881,024 versus 186,646,528 bytes move/streamed; warm 232,783,872/186,646,528.
Explicit bound saves 44,620,736 bytes (42.554 MiB). Eight-mode samples 12,582,912/
8,388,608 save 2,932,960 explicit bytes. Granularity prevents treating these as
precise owned-allocation peaks. The ownership control changes the interpretation:
keep resident when it fits, consider streaming for capacity, and port explicit
return ownership separately instead of attributing copy savings to streaming.

Final uncommitted outputs-02 was regenerated after narrowing recorded source hashes
to executable sources; hashes match current code. outputs-01 remains a superseded
preliminary capture: its uncommitted two-route launcher evolved before commit, so
use final three-route source/outputs-02 for reproduction and decisions. All raw
work is retained and this limitation is explicit. No production policy changed.

Next preserve the copy-return baseline and qualify explicit return move in production,
then independently measure placement/host peaks before promoting a capacity selector.
Goal active; input/slab streaming, broader batching/cutoff and GH200-class qualification
remain. All verification handles are terminal. Code/data and plan/log commits separate,
no push/install/supervisor changes, unrelated work retained. Evidence: output_sectors/
thinkstationP620_RTXA4000/{outputs-01,outputs-02}; /tmp/xca-bs-output-bench-{configure,
build,campaign,memcheck}.log, corresponding move configure/build/campaign logs and
library configure/build logs with /tmp/xca-bs-output-bench-library-control.csv.

## 2026-10-14 — Production self-energy return ownership

Previous turn was progress: ownership-controlled complete-call measurements changed
the streaming decision. Production compute_self_energy_range now moves its host GF
vector into the owning block_gf constructor instead of copying it. No device arithmetic,
device bounds, method signatures or Python API change. The experiment generator
normalizes return syntax for its streaming template and explicitly emits copy/move
controls. Benchmark resident now names CopyOutputEvaluator; resident_move uses the
actual production DiagramEvaluator. Earlier baselines and raw work remain available.

CPU-only configure/library compilation passes. Main targeted build and four CTests
pass in 39.73 s: production evaluator/admission, private complete-output controls and
six-method Python integration. Native-disabled targeted build/bindings and three
matching CTests pass in 24.47 s, flags remain OFF. Focused three-route memcheck gives
zero errors/leaks. GPU verification stages are sequential; performance starts only
after all builders/tests/profilers are terminal. Binding regeneration is unnecessary
because no exported signature changed. Full suites and two-rank MPI remain the prior
compact checkpoint, not newly claimed for this one-line ownership change.

Final production outputs-03: seven processes / 147 rows match exact outputs/counts;
all recorded source hashes match current executable inputs. Changed-G copy/production/
streamed medians: eight-mode 21.375454/18.823287/18.766029 ms, ten-mode 207.280464/
162.649520/173.526853 ms. Production reduces local API time 11.94%/21.53% versus the
preserved copying control. This avoids attributing ownership savings to sector
placement and is not a whole-Solver or GH200 speedup claim. Device envelopes unchanged.

Goal active. Next investigate/select a production memory placement policy, retaining
resident execution when it fits and requiring complete-output/host accounting for
streaming; remaining input/slab, many-small batching, cutoff and target gates stay
open. All handles terminal. Code, measurement and plan/log commits stay separate;
no install/push/supervisor changes; unrelated work preserved. Logs /tmp/xca-bs-return-
move-{configure,build,test,cpu-configure,cpu-build,library-configure,library-build,
library-test,memcheck,campaign}.log; output_sectors/thinkstationP620_RTXA4000/outputs-03.

## 2026-10-14 — Production C++ output placement

Previous goal turn was progress: explicit return ownership was promoted and measured.
Added OutputPlacement::{Resident,HostSectors} as a C++ constructor option, followed
by host output payload budget (512 MiB default). Resident remains default. HostSectors
keeps exactly one output sector on GPU and returns every host sector, with unchanged
GPU accumulation/path/cutoff order. Inputs remain resident. SE estimates substitute
one maximum tile for one full output function; NCA still removes inactive DLR scratch.
SPGF/correlator estimates/implementation remain unchanged. Host payload admission is
before upload/allocation, including empty ranges, and excludes caller G, metadata,
allocator overhead and a transient sector download. Invalid placement rejects.

Preserved pre-placement resident source/header snapshots inside output_sectors so
private copy/move/stream baselines remain intact. Updated scalar visitor/accessor and
prefix getter transformations for production placement; scalar experiments explicitly
require Resident rather than silently ignoring HostSectors in arena shortcuts.
New HostOutputControl runs all eight reusable observable/API controls with the actual
production HostSectors option, including dynamics, both correlator statistics and
range/cutoff behavior. New production admission control checks three partitions,
Library/Reference and available Ordered/Mapped, NCA/higher-order bounds, exact full
outputs, tight accept/reject, changed G/masks, empty outputs and invalid enum.

Main initial six focused CTests pass in 65.15 s. Binary listing showed the newly
added admission control was absent because its source changed after that target's
compilation; those results were not treated as new-policy evidence. Rebuilt admission
and HostOutputControl: three admission tests plus eight forced-policy controls pass
in two CTests / 9.44 s. Added recovery proof that host-output rejection does not poison
a subsequent SPGF call, then rebuilt admission and scalar guard: two CTests pass
10.61 s. Native-disabled four focused CTests pass 25.77 s; final updated admission
rerun passes 2.23 s. Both native flags remain OFF. CPU configure/library build passes.
Production CompleteHostOutputPlacement memcheck passes in 11.757 s, zero errors/leaks.
GPU verification stages sequential. Full suites/MPI remain prior compact checkpoints;
these are focused C++ placement controls, not newly claimed full suites.

C++ policy is qualified and explicit; Python facade and Solver/MPI selection are
still pending and documented. No new Python arguments or automatic post-submission
fallback are claimed. Next integrate placement and host budget through the consolidated
facade/configuration, archives/deepcopy and collective error handling. Capacity still
needs input/slab work and GH200-class validation; local RTX A4000 is a proxy. Goal
active. All handles terminal, all experiments/user work retained, code and plan/log
commits separate, no push/install/supervisor changes. Logs /tmp/xca-bs-placement-
{configure,build,list,test,final-configure,final-build,final-test,cpu-configure,cpu-build,
library-configure,library-build,library-final-configure,library-final-build,library-test,
memcheck,recovery-configure,recovery-build,recovery-test,library-recovery-build,
library-recovery-test}.log. No timing experiment ran during this qualification.

## 2026-10-14 — Python and MPI output placement

Previous turn was progress: production C++ HostSectors policy was qualified and
committed. Consolidated facade static/dynamic constructors now accept output_placement
and host_output_budget, validating placement before export. Updated explicit real/
complex template instantiations, regenerated only _triqs_xca_cuda bindings and applied
the CUDA guard. Solver adds block_sparse_output_placement and block_sparse_host_output_
budget, resident/512 MiB defaults. GPU helper validates placement before module load
and host budget as nonnegative integer (bool rejected); CPU stores preferences without
CUDA import. Both attributes survive deepcopy/HDF and old-archive reconstruction.
Runtime mutation validation follows existing collective initialization handling.

Added Python static/dynamic HostSectors equivalence, smaller bound, rank-local zero
budget collective evaluation failure, invalid placement initialization failure and
recovery controls. Updated archive/default/validation checks. Initial wrapper generation
failed until explicit template-instantiation signatures included the new arguments.
First tests found inconsistent bool exception and a recovery fixture calling solve,
which changed physical inputs; standardized ValueError and used the collective init
entry point without changing inputs. Build-tree Python files are copies: one stale
rerun was discarded, then all three trees were reconfigured before final checks.

Final main Python CTest passes seven methods in 26.44 s; native-disabled seven pass
13.06 s, both native flags remain OFF. CPU-only metadata/archive control passes .40 s,
six GPU methods skipped. After these processes terminate, two-rank run passes seven
methods each in 118.970/118.966 s, using CTest JSON environment and OMP/BLAS threads one.
Failures/recovery are collective and no rank falls back to CPU evaluation. Known X11
startup messages are not tool approval rejection. All handles terminal. No performance
claim, new full-suite claim or target-hardware inference from these focused tests.

Goal active. Placement is now operational through C++/Python/MPI with complete outputs;
inputs remain resident. Next qualify larger placement capacity/host peaks and audit
remaining batching/cutoff/target gates. GH200-class is target; A4000 a development proxy.
All baselines/experiments/user work preserved. Code/bindings/tests and plan/log commits
separate; no install/push/supervisor changes. Logs /tmp/xca-bs-placement-wrap.log,
/tmp/xca-bs-placement-python-{build,test,refresh,cpu-configure,cpu-test,library-build,
library-configure,library-test,mpi}.log. API details in doc/python_api.rst and Solver
docstring; generated constructors include the two new keywords.

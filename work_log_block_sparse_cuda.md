# Block-sparse CUDA work log

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

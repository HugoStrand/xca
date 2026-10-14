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

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

# Dense CUDA implementation work log

## Current checkpoint — 2026-10-10

Branch: `dev_cuda`. Session: `01a11fc7-86f7-7130-9dd9-8a073f35fc9f`.
Phase: ordinary self-energy and by-pairs CUDA traversal verified.
Next: dynamic-interaction self-energy validation, then correlator implementation.

## Objective and working rules

Implement `plan_dense_cuda.md` within its C++ evaluator scope. Preserve CPU support
and compute/count semantics, reuse existing C++ comparisons, validate device/DLR
primitives before full traversal, and deliver residency/allocation evidence and
end-to-end benchmarks. Use GPT-6.1 Sol / medium / standard speed by default.

Commit verified implementation/test milestones on `dev_cuda`, without pushing.
Commit `plan_dense_cuda.md` and this log separately from code, tests and supervisor
changes. Read this checkpoint, current AGENTS.md, and Git status on every resumption.
Update evidence, failures and the precise next action after each milestone.

Preserve pre-existing user work: modified `.gitignore`; untracked `AGENTS.md`,
`AGENTS.md.bak-20261009`, `test/c++/test_nda_cuda_smoke.cpp`, and
`test/python/exp_order_conv/`. Do not stage these with implementation work.

## Completed: custom quota-reset supervisor

- Added a Python standard-library Unix WebSocket/JSON-RPC client and supervisor in
  `tools/codex_supervisor/`, plus a detached launcher and operational README.
- Uses the existing shared app-server, never a competing `codex exec` worker.
  Checks goal identity, repository, branch, live activity and backend included-usage
  permission. Reset timestamps guide polling only. Honors pauses, blockers, budgets,
  workspace restrictions, non-quota failures and approval waits.
- Journals uncertain turn starts before submission; refuses blind retries after a
  lost response. OS locking prevents duplicate supervisors for this runtime.
- Installed machine-specific config/state/logs under ignored `cbuild-supervisor/`.
  Dedicated tmux server `xca-cuda-supervisor`, session `dense_cuda`, survives logout.
  Installed and verified one user-crontab `@reboot` entry marked
  `xca-dense-cuda-supervisor`; existing crontab entries were preserved.
- Launcher starts the managed daemon only if absent and does not restart live work.
  Initial local `PAUSE` was removed after persistent goal activation.

Validation: all 23 tests passed via
`python3 -m unittest discover -s tools/codex_supervisor/tests -v`.
Tests include actual Unix socket/WebSocket framing and a simulated quota exhaustion,
elapsed reset with permission still denied, explicit backend recovery, and exactly
one continuation. Also tested weekly/workspace limits, pauses, active races,
lost-response restart, lock contention and idempotent/quoted launch.

Live checks against the current daemon (0.162.1; CLI 0.162.0): current thread is
active, model `gpt-6.1-sol`, effort `medium`; `ordinaryUsageAllowed` is true;
primary window is 300 minutes and secondary window is 10080 minutes. Thread metadata,
goal reads and latest-turn pagination work. Verified detached process launch twice
without creating a duplicate session.

Limits of verification: did not deliberately exhaust the real account quota or
reboot ts. The next natural quota reset validates real backend recovery. Unknown
permission, authentication/service failures and ambiguous starts fail closed.
See the supervisor README for pause/stop/restart controls and these limitations.

The normal command sandbox fails at startup with the recorded bubblewrap loopback
error. Escalated commands work through configured automatic approval review. One
large file-write review timed out; retrying the write succeeded. No approval rejection
or bypass was required.

## Environment and baseline

- Verified the CUDA NDA target exports `NDA_HAVE_CUDA` and `NDA_HAVE_CUTENSOR`;
  CUDA `libnda_c.so` has a `NEEDED` entry for `libcutensor.so.2` and the cuTENSOR
  library directory in RUNPATH. Revised the plan's outdated absence assumption.
- Use `~/env/xca-cpu.sh` / `~/env/xca-cuda.sh` for every corresponding build/test.
  Reconfigure in place after adding source files; preserve cached dependencies.
- Prior clean baseline on 2026-10-09: CPU 61/61 CTest entries passed (53.83 s),
  CUDA 61/61 passed (52.35 s), with real GPU smoke cases and adapol `d3c4ab7`.
  Those results precede this supervisor setup and are not new evaluator validation.

## Goal activation and commit record

The persistent goal is active with the exact objective bound in supervisor config;
no explicit goal token budget was requested or set. Removed the initial local PAUSE
and verified live `state.json` transitions to `thread_active`. The detached
supervisor observes this current turn without submitting another continuation.

Separate initial commits:

- `c584bb5`: supervisor code, tests and operational README.
- `d33339d`: plan only, including verified cuTENSOR availability.
- `b63fd84`: initial work log only.

This activation checkpoint is also committed as a log-only change.

## Completed: initial device primitive gate and CPU baseline

Previous goal turn classification: progress (supervisor installed, goal activated,
checkpoint committed). This turn also changes authoritative state through verified
CUDA code and reproducible benchmark artifacts; the overall goal remains active.

Implementation commits:

- `1aba386`: `dense/cuda/device_ops.{hpp,cpp}` and five actual GPU test cases.
- `82d834a`: bounded C++ benchmark target and baseline artifacts.
- `d5502bf`: CSV line-ending normalization found by the whitespace check.

`DeviceOps` supplies complex-double matched/repeated-operator batched GEMMs,
reusable unified pointer tables, device copy/zero/scaling/AXPY, time-row DGMM,
C/F GEAM packing and non-conjugating flattened trace GEMM. It uses an owned native
cuBLAS handle on the default stream, explicit completion, checked dimensions/byte
products, contiguous layout checks and output-alias rejection. No work tensors or
batch metadata are allocated by these methods. This is a primitive layer, not yet
a full evaluator or a measured inner-loop residency guarantee.

Validation:

- Reconfigured both trees in place with their environment switchers; preserved
  cached dependencies and local configure scripts.
- Both CPU and CUDA built `triqs_xca_c`, `dense_cuda_device_ops`, and
  `dense_cuda_benchmark`. CPU binding regeneration produced no tracked changes.
- CUDA `dense/cuda/device_ops` CTest passed; all five GoogleTests actually ran on
  the GPU. Tests exercise complex operand order, metadata reuse/accumulation,
  rectangular C/F layout conversion without conjugation, complex trace semantics,
  weighting, alias/strided input rejection, checked sizes and post-exception reuse.
- Compute Sanitizer memcheck passed all five cases with **0 errors**.
- CPU counterpart compiled and its CTest passed while running zero GoogleTests
  (the no-CUDA case is deliberately disabled); this proves build compatibility,
  not GPU behavior. Full suite runs remain a later evaluator acceptance gate.
- Initial test failures were corrected test issues: an F-layout transpose cannot
  be passed to a C-layout view at compile time, so runtime stride rejection uses a
  genuine strided C-order view; NDA near-comparison uses strict `< tolerance`, so
  zero-tolerance exact-copy assertions were replaced with exact zero-difference
  assertions. Device results themselves matched in those failures.

Evidence logs: `/tmp/xca-{cpu,cuda}-primitives-{configure,build,test}.log`,
`/tmp/xca-cuda-primitives-sanitizer.log`. Temporary logs can disappear; source,
tests and benchmark CSV/README are tracked. Final formatted sources were built in
both trees and the focused GPU test was rerun successfully.

CPU baseline: `benchmarks/dense_cuda/cpu_baseline.csv` and accompanying README.
Ran N=2/4/16/32/64, rank 26, two poles, orders 1–3, at most eight selected indices,
three warm repetitions, one BLAS/OpenMP thread, on AMD Threadripper PRO 3945WX.
The synthetic dense Fourier-rotated propagator and occupation-basis operators are
constructed outside call timings. The whole API and selected isolated operations
are timed; no exact whole-call phase fractions or CUDA speedups are claimed.

At N=32, warm current/hoisted convolution is 7.12/1.77 ms and one left vertex is
0.157 ms. At N=64, corresponding values are 30.73/10.56 ms and 1.04 ms. Hoisting the
two invariant rank transforms matches the current cppdlr convolution exactly in
all these runs. This measurable CPU improvement must be distinguished from GPU
acceleration in the final performance report.

## Completed: resident device DLR layer

Previous goal turn classification: progress (verified primitive code, benchmark
artifacts and separate checkpoint committed). This turn adds verified DLR code;
no blocker is present and the complete evaluator goal remains active.

Commit `4d4a0bc` adds `dense/cuda/dlr_ops.{hpp,cpp}`, a checked rank-axis GEMM in
DeviceOps and four GPU test cases in `test/c++/dense/cuda/dlr_ops.cpp`.

DlrOps owns immutable device copies of cached LU/pivots, C/H/Q and the reflection
matrix. It preserves the cached LU's C-layout/NDA getrs transpose convention,
uses device GEAM to pack/unpack a persistent Fortran RHS, checks solver status,
and reuses four convolution tensors. Hilbert transforms are computed once each;
three batched products build P/R, followed by two accumulating rank GEMMs with beta.
Reflection is a value-to-value matrix constructed by host `reflect(identity)` once.
No explicit inverse replaces the LU solve.

NDA source review confirms its solver handle and unified one-word status storage
are lazy process-shared state. Construction warms those and the shared BLAS handle
using existing buffers, keeping initialization ahead of traversal. Solver
synchronization is required and checked at construction and at every solve; the
backend never silently changes the thread-local setting. The initial API requires
serialized NDA device solver use, including other NDA users in the same process.
Final allocation/residency tracing is still required; library internals are not
assumed allocation-free merely because XCA reuses its buffers.

Validation:

- Reconfigured in place and built `dense_cuda_dlr_ops` in both CPU/CUDA trees.
- GPU CTest passed with all four GoogleTests executed; Compute Sanitizer memcheck
  passed all four with **0 errors**, including the failed-constructor resource path.
- The small rank-3 well-conditioned system checks coefficients against both the
  known coefficients and cppdlr, including in-place conversion, at 1e-12.
- Real mesh tests cover lambda 10/40/120, eps 1e-10/1e-12, symmetric and non-symmetric
  meshes, beta 0.5/2/7, complex noncommuting matrix inputs, reconstruction, reflection
  and reflection twice, coefficient convolution and the full two-solves→convolve
  edge pipeline. Comparisons pass at 1e-12 without a GPU-specific tolerance.
- Additional checks cover cancelling coefficient inputs, changed input at the same
  allocation, reusable scratch, convolution output aliasing either input, invalid
  shapes/beta/transform aliasing and preservation of the caller's synchronization
  setting after rejection. The solver's disabled-sync constructor/call contract is
  exercised explicitly.
- CPU test compiles/runs its deliberately disabled no-CUDA branch; it executes zero
  GoogleTests and is compatibility evidence, not numerical device validation.

Evidence: `/tmp/xca-{cpu,cuda}-dlr-{configure,build,test}.log` and
`/tmp/xca-cuda-dlr-sanitizer.log`. Final sources were rebuilt in both configurations;
GPU tests and sanitizer were rerun after the constructor warmup/contract changes.
No tracked binding regeneration changes were produced.

## Completed: ordinary self-energy and paired traversal

Commit `e8138dc` adds `dense/cuda/diagram_evaluator.{hpp,cpp}`, updates the device
convolution grouping, and extends five existing C++ test seams. The preceding goal
turn revalidated supervisor/status without changing implementation; this turn makes
concrete implementation progress. No blocker is present; the full goal remains active.

The independent CUDA class owns frozen host mesh/hybridization/FSet setup and device
operator, kernel and hybridization tables. It does not inherit CPU scratch. Checked
workspace arithmetic runs before device workspace allocation, including BLAS integer
limits, aggregate complex-array bytes, pivots and persistent pointer tables. Native
allocators report allocation failures; final live-memory measurement remains pending.

Compute/count interfaces use long indices. All/single/vector self-energy and by-pairs
interfaces preserve duplicate indices, explicit pair seeds and direction, empty
subsets and reset of the accumulated sum after a prior failure. The all-pairs path
selects fb(0)=0 representatives. Each public call packs/uploads the full propagator
once and downloads the accumulated result once. All traversal GEMMs, DLR solves and
convolutions operate on persistent device tensors.

Special-vertex backward weights use (kap,mu), forward weights (mu,kap); parity is
recomputed after setting both orbitals. Prefactor and diagram-order/direction signs
follow the CPU implementation. Edge kernels are applied sequentially using uploaded
pole kernels and swapped device scratch, preserving G*k1*k2 arithmetic and removing
the initial small host edge-weight staging. Convolution now uses separate product
addition and final beta scaling in cppdlr's order, reusing consumed scratch. No
explicit inverse or reduced precision was introduced.

Both real/complex atom_diag constructors and the low-level FSet constructor compile;
the dynamic-interaction constructor uploads extended coefficients/operators but its
numerical gate is the next task. Correlator and single-particle methods are not yet
implemented. Runtime tracing has not yet established library-internal allocation
or transfer behavior; resident source-level workspace alone is not final evidence.

Validation on final sources:

- CPU and CUDA targeted builds passed. Both focused gates passed all seven CTest
  entries: one-/two-fermion analytic fixtures, complex unitary invariance,
  flat-index range, propagator layout, DeviceOps and DlrOps. CPU primitive entries
  deliberately execute no GoogleTests; CUDA tests actually execute GPU work.
- Added 36 CUDA comparison points to existing analytic self-energy tests, including
  orders 1–3 and ordinary/by-pairs variants, at the original test tolerances.
- Existing complex-rotation seam validates ordinary/by-pairs self-energy covariance
  and agreement with CPU for both original and rotated off-diagonal problems.
- Subset tests cover duplicates, either pair direction, count equality, a failed
  vector after a valid prefix, next-call reuse, empty input, changed topology and
  changed propagator data at the same host allocation.
- Existing wrong-layout checks now exercise all six CUDA self-energy overloads;
  negative/out-of-range index checks preserve the Backbone exception contract.
- The order-six count is 49,766,400,000. Twelve nearby nonzero indices beyond INT_MAX,
  plus vector and pair interfaces, are explicitly evaluated on the GPU.
- Compute Sanitizer memcheck passed three focused traversal GoogleTests and all four
  revised DLR GoogleTests with **0 errors**. No tracked binding changes appeared.

Numerical investigation and documented exception: the original order-six relative
1e-13 CPU/block-sparse threshold failed the new CUDA comparison. Its ten edges chain
twenty LU solves. Initial composed edge weights produced 1.12e-13 relative difference
at the original selected index. Matching cppdlr's separate addition/final scaling did
not change that result. Sequential resident kernel multiplication changed the selected
error to 1.72e-13, demonstrating arithmetic sensitivity. Measuring twelve nearby long
indices found 1.4e-13–1.3e-12 relative differences, all below 4.4e-20 absolute. Following
the plan's numerical policy, only these order-six CUDA assertions use a documented
3e-12 relative bound (below this mesh's eps=1e-10). CPU/block-sparse assertions remain
1e-13, and the analytic/primitive/covariance assertions retain their original bounds.
This is evidence of accumulated CPU/CUDA floating-point differences, not an isolated
proof that cuSOLVER alone causes them. No tolerance was silently changed.

Logs: `/tmp/xca-{cpu,cuda}-evaluator-final-{build,test}.log`,
`/tmp/xca-cpu-evaluator-final-refresh.log`, `/tmp/xca-cuda-evaluator-sanitizer.log`,
`/tmp/xca-cuda-dlr-grouping-sanitizer.log`, and
`/tmp/xca-cuda-order-six-roundoff.log`. Temporary logs may disappear; test comments
preserve the measured range and bound. Final full-suite runs and performance tracing
remain acceptance requirements after all compute paths are implemented.

## Next action

Validate dynamic-interaction self-energy using the existing `dense/dynint_parity.cpp`
and `dynint_spin_rotation.cpp` seams. The constructor already snapshots the extended
hybridization/FSet; verify mixed fermionic/bosonic parity and all/single/vector/pairs
behavior before treating that path as complete. Test the low-level constructor and
real atom_diag constructor numerically, plus non-contiguous host input at the API
boundary where the current source packs it.

Then port correlator traversal into the same CUDA class: right vertex/edge ordering,
reflection, endpoint operators, non-conjugating traces, single-particle all/single/
vector/count interfaces and one-time correlator with explicit statistics. Read the CPU
correlator code before writing it. Vertex sign=-1 selects the opposite pole kernel;
it is not an extra scalar minus. Reuse device scratch and upload per-call endpoint
operator tables outside traversal. Extend the existing analytic/unitary/statistics/
dynamic/long-index seams as methods become available, without a universal framework.

Complete final CPU/CUDA C++ suites, actual GPU sanitizer checks, allocation/residency
traces and end-to-end benchmarks before claiming the goal achieved. Keep plan/log
commits separate from implementation/tests. Supervisor setup is complete; do not
recreate the goal or restart the shared server. Preserve unrelated user work and do
not push.

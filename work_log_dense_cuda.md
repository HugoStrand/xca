# Dense CUDA implementation work log

## Current checkpoint — 2026-10-10

Branch: `dev_cuda`. Session: `01a11fc7-86f7-7130-9dd9-8a073f35fc9f`.
Phase: all planned C++ compute paths implemented and numerically verified.
Next: allocation/residency tracing, checked workspace inventory and end-to-end benchmarks.

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

## Completed: correlators and dynamic-interaction numerical gate

Commit `4327fe4` adds CUDA single-particle all/single/vector/count methods, one-time
correlators for real/complex atom_diag and both explicit-statistics eval_correlator
wrappers. Previous goal turn: progress (self-energy/by-pairs code and checkpoint).
This turn is also progress; no blocker exists. The full goal remains active because
residency/allocation and benchmark deliverables are not yet achieved.

The right traversal multiplies on the right, selects the opposite pole-kernel cache
without an extra scalar minus, and convolves U coefficients before G coefficients.
Reflection, prefactor and external-statistics sentinels match the host evaluator.
Endpoints contract as U*(mu*T), followed by a packed, non-conjugating trace GEMM.
Fermion parity uses the shared operator_statistics classifier. Dynamic external
single-particle legs exclude the appended interaction operators.

Endpoint/operator/output device buffers persist across calls and resize only when
endpoint counts change. Host operator assembly and uploads happen before flat-index
traversal. Trace accumulation stays on device and returns only the completed result.
Empty endpoint sets retain flat-index validation; empty operator lists/zero or mixed
parity follow the classifier's exceptions. Existing shape/AD constraints are retained.
Constructor workspace arithmetic and endpoint byte/BLAS checks exist; the final
aggregate live-workspace inventory and tracing of library-internal allocations remain
necessary. Do not infer the allocation gate from source reuse alone.

Verified on this machine:

- Complete CPU build and all **33/33 C++ CTest entries passed** (13.84 s).
- Complete CUDA build and all **33/33 C++ entries passed** (22.40 s), including
  actual GPU smoke/primitives and the new evaluator comparisons. Python tests were
  excluded as specified by the C++ port scope; the full builds still build bindings.
- Added 18 single-particle analytic comparison points for orders 1–3 to the existing
  one-/two-fermion tests at their original tolerances. Their subset cases now cover
  duplicate indices, failed-prefix reuse, empty subsets and long counts.
- Registered CUDA alongside dense/block-sparse in the complex unitary seam and made
  cross-evaluator comparisons iterate over all entries. Both self-energy and spgf
  covariance pass; ordinary/by-pairs self-energy checks are retained.
- Spin-rotation tests compare CUDA self-energy/spgf with dense results for ladder,
  Cartesian (including imaginary S_y) and reduced/no-interaction forms, with real
  and complex atom_diag. The new dynint-parity test covers every self-energy/pairs
  overload, every spgf overload, counts, density correlators and failed-prefix reuse.
- Existing statistics helper is narrowly templated for immutable CUDA metadata.
  Forced bosonic/fermionic single-index and all-index wrappers agree with CPU;
  automatically classified spin-flip, density and fermion correlators retain the
  original 1e-13*max(1,scale) gate. Mixed parity rejection/all-odd acceptance pass.
- Added real/low-level constructor comparisons, genuinely strided propagator input
  (last stride 2), changed data at the same storage, changing endpoint count and
  changed endpoint content with unchanged shape. Empty endpoints, zero statistics,
  wrong AD dimension, multi-subspace AD and dynamic constructor rejection pass.
- Long spgf and one-time correlator indices beyond INT_MAX pass using the previously
  documented order-six 3e-12 CUDA bound; no additional tolerances were changed.
- Final negative-test additions were rebuilt and verified in focused CPU/CUDA runs
  after the full suites; final wrapper additions were also rebuilt in both trees.
- Compute Sanitizer memcheck: strided/constructor/endpoint case (1 GoogleTest),
  dynamic-interface/reuse/constructor-exception case (1), and statistics (3), all
  **0 errors**. The latter checks both explicit-statistics wrapper overloads.
- No tracked binding regeneration changes; unrelated user work remains unchanged.

One new test initially failed to compile because an injected fermion-block check
referenced the boson result from another scope. Corrected that test before the
reported passing gates. No implementation numerical failure or new tolerance change
was required by this milestone.

Logs: `/tmp/xca-{cpu,cuda}-correlator-final-{build,test}.log`,
`/tmp/xca-{cpu,cuda}-correlator-rejections-{build,test}.log`,
`/tmp/xca-{cpu,cuda}-correlator-wrapper-{build,test}.log`, and
`/tmp/xca-cuda-correlator-{layout,dynint,statistics}-sanitizer.log`.
Some first-pass logs use shorter names. Temporary logs may disappear; tests and this
checkpoint preserve the evidence and measured full-suite times.

## Next action

Extend `examples/dense_cuda_benchmark.cpp` to compare CPU and CUDA with identical
inputs/subsets. Include multiple DLR ranks and N=2/4/16/32/64/128 where feasible,
orders 1–3, construction, first/warm API timings including transfers, device-phase
CUDA event measurements on the actual default stream and owned/live memory evidence.
Keep the CPU hoisted-transform baseline distinct from GPU offload. Compare returned
results in the benchmark; do not time unchecked work or claim speedup from fixtures.

`nsys` is available at `/usr/local/bin/nsys`, and CUPTI headers are under
`/usr/local/cuda/targets/x86_64-linux/include`. Use Nsight Systems CUDA traces with
explicit boundary/traversal ranges to measure allocations, transfer sizes, launch
counts and GEMM/solve time, proving no large tensor copies or device allocations in
inner evaluation loops. Warm library paths outside traversal or fix repeated library
allocations if the evidence shows them. Extend the checked workspace inventory to
include persistent endpoint buffers/peak-live resize behavior and clear allocation
failure context. Inspect profiling tool help locally before choosing commands.

Review final implementation against every acceptance requirement in the plan. Run
required final CPU/CUDA C++ gates after material changes and targeted sanitizer checks
when new memory paths justify them. Deliver tracked benchmark CSVs/report and trace
summaries with workload/GPU/thread metadata. Do not mark the goal complete until
all allocation/residency/performance evidence is present and reviewed.

Supervisor setup is complete; do not recreate the goal or restart the shared server.
Keep plan/log commits separate from implementation/tests/benchmarks. Preserve unrelated
user work and do not push.

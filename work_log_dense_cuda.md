# Dense CUDA implementation work log

## Current checkpoint — 2026-10-10

Branch: `dev_cuda`. Session: `01a11fc7-86f7-7130-9dd9-8a073f35fc9f`.
Phase: initial device primitives and bounded CPU baseline verified; next is device DLR operations.
The CUDA evaluator itself is not yet implemented.

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

## Next action

Implement `dense/cuda/dlr_ops.{hpp,cpp}` and focused GPU/CPU-reference tests before
integrating diagram traversal. Upload cached complex LU factors/pivots, C/H/Q
matrices and a reflection matrix once, reuse a Fortran-order RHS and convolution
scratch, and use the validated DeviceOps operations. Preserve the cached LU's
layout/transpose convention rather than treating its logical entries as a fresh
unfactored matrix. Require NDA solver synchronization before `getrs` because it
reads unified `info` immediately; do not disable or silently change shared settings.

Construct H/Q using the installed cppdlr time-ordered formulas, including its
relative-node branch at zero. Build the value-to-value reflection matrix by applying
host `itops.reflect` to an identity matrix. Validate transforms/reconstruction,
reflection and noncommuting complex matrix convolution across beta/rank/symmetric
mesh choices and cancellation-sensitive inputs. Investigate LU coefficient
sensitivity using reconstruction/residual evidence; do not weaken CPU assertions.
Then port ordinary self-energy constructor/compute paths using existing test seams.

Supervisor setup is complete; do not recreate the goal or restart the shared
daemon. Continue committing verified implementation/tests on dev_cuda and this
checkpoint separately; preserve the listed unrelated user changes.

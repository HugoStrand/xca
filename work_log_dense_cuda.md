# Dense CUDA implementation work log

## Current checkpoint — 2026-10-10

Branch: `dev_cuda`. Session: `01a11fc7-86f7-7130-9dd9-8a073f35fc9f`.
Phase: supervisor setup verified; persistent goal activation follows this checkpoint.
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
  Initial local `PAUSE` file holds custom recovery until goal activation.

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

## Next action

Activate the persistent implementation goal with the exact objective in local
supervisor config, remove its initial local PAUSE, verify observation of this active
thread, and commit this activation as a log-only checkpoint. Then start phase 1:
inspect dense evaluator/NDA interfaces and existing C++ test seams, profile a bounded
CPU workload, and implement/validate the device primitives required by DLR operations.
Keep plan/log commits separate from code throughout.

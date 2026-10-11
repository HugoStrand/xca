# Dense CUDA work supervisor

This supervisor watches one explicitly authorized persistent Codex goal on the
existing shared app-server. It does not run `codex exec`, create another worker
server, change permission profiles, buy credits, consume earned resets, or switch
accounts. Python's standard library implements the local Unix WebSocket transport;
there are no additional packages or TCP listeners.

## Recovery policy

- Match the configured repository, `dev_cuda` branch, thread ID and exact objective.
- Leave active turns, including approval/question waits, alone.
- Respect `paused`, `blocked`, `budgetLimited`, `complete`, and cleared/replaced goals.
- Require a fresh `account/rateLimits/read` response with
  `ordinaryUsageAllowed: true`. Null/missing permission is unavailable, not recovery.
  Check the overall and applicable model buckets for contradictory exhausted
  windows or workspace/spend-control blocks. Reset times only set polling delays.
- After 120 seconds of stable idle state, recover `usageLimited` with automatic
  mutation provenance and start a continuation at GPT-6.1 Sol / medium / standard
  speed. Active goals also have an idle fallback if built-in continuation stalls.
- A terminal quota failure can leave the loaded thread in `systemError`. Treat
  that state as recoverable only when its latest turn is failed with
  `usageLimitExceeded` or `rateLimitExceeded`, and the backend permits usage.
  Recheck the same failed turn immediately before starting, including after goal
  recovery. Other error states remain blocked. No thread reload is necessary.
- Preserve the resumed thread's existing tools, environments and permission review.
  Recheck live status immediately before starting; the shared server serializes
  turns on this thread, including a racing UI request. A final race can add input
  to that same turn, but cannot create a separate concurrent evaluator worker.
- Interruptions and failures other than quota errors require manual intervention.
  Unknown start outcomes are journaled before submission and reconciled with turn
  history. If no new turn is visible, halt rather than risk a duplicate retry.
- Bound connection backoff and quota polling to 15 minutes. Completion exits the
  supervisor; user pause and blockers remain observable without being overridden.

## Local installation on ts

Machine-specific configuration and all runtime state are in ignored build folders;
the current optimization goal uses `cbuild-supervisor-small/exploration/` (the
completed original port used `cbuild-supervisor/`). Code and tests are tracked here. A dedicated tmux server
(`xca-cuda-supervisor`, session `dense_cuda`) survives SSH logout. A user crontab
`@reboot` entry runs `launch.py`, which starts the managed Codex daemon if absent
and creates the detached supervisor session if absent. It never restarts a running
daemon. Boot recovery requires cron, the home/repository paths, and usable existing
Codex authentication. Hardware/network/service failures can still require repair.

From the repository root:

```bash
python3 tools/codex_supervisor/launch.py --config "$PWD/cbuild-supervisor-small/exploration/config.json"
cat cbuild-supervisor-small/exploration/state.json
tail -n 20 cbuild-supervisor-small/exploration/supervisor.log
```

To pause the **goal**, use `/goal pause` in the session. To disable only custom
restarts, create `cbuild-supervisor-small/exploration/PAUSE`; remove it to enable observation again.
That file does not interrupt an already running turn or pause built-in Goal
continuation. To stop the detached supervisor itself:

```bash
tmux -L xca-cuda-supervisor kill-session -t dense_cuda
```

Remove the crontab line marked `xca-dense-cuda-supervisor` to uninstall boot launch.
For `uncertain_start` / `halted_manual_review`, inspect the session and reconcile
whether the request ran before clearing `halted`/`pending` in local state and
restarting. Never clear those fields just to force another request.

A one-tick read-only check (stop the running supervisor first because the same
runtime lock protects all invocations):

```bash
python3 tools/codex_supervisor/supervisor.py \
  --config "$PWD/cbuild-supervisor-small/exploration/config.json" --once --dry-run
```

## Validation

```bash
python3 -m unittest discover -s tools/codex_supervisor/tests -v
```

Tests cover primary/weekly quotas, workspace credits, explicit permission recovery,
user pauses/blocks/budgets, active-turn races, objective/branch guards, lost start
responses across restart, atomic state, OS process locks, launcher idempotency and
quoting, WebSocket framing/fragmentation/pings, and end-to-end simulated reset.
Live checks read the current session and included-usage permission. No real account
quota is deliberately exhausted; the next natural reset remains the production
validation of backend recovery. Boot registration is checked without rebooting ts.

`state.json` records each poll's time, runtime thread status and latest inspected
turn ID/status/error code. The log prints action changes only; an unchanged log
does not mean the watcher stopped. A quota reset is simulated with a thread that
remains `systemError` until `turn/start`, matching the observed missed-restart case.

Protocol reference: [official Codex app-server documentation](https://learn.chatgpt.com/docs/app-server).
The generated protocol schema for the installed CLI also documents
`ordinaryUsageAllowed` and automatic goal-mutation provenance. The transport is
experimental: incompatible upgrades fail closed and require revalidation.

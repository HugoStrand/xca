# Small-matrix CUDA benchmark and optimization work log

## Current checkpoint — 2026-10-11

Preparation goal active: build/validate a small reproducible high-order benchmark,
then write/review the optimization plan. Only then switch this thread's subsequent
turns to `gpt-6.1-sol` / medium and create the separately authorized exploration goal.
Current preparation model is `gpt-6-astra` / high. User requested both persistent goals.

Branch dev_cuda. Production CPU/CUDA evaluators at baseline commit064b64e remain
unchanged. Keep experiments in individual folders under experiments/dense_cuda_small/;
never replace the production baseline. Commit code separately from plan/log, no push.
Preserve modified .gitignore and untracked AGENTS.md, AGENTS.md.bak-20261009,
test/c++/test_nda_cuda_smoke.cpp and test/python/exp_order_conv/.

## In progress

New shared harness `benchmarks/dense_cuda_small/harness.hpp`, executable
`examples/dense_cuda_small_benchmark.cpp`, runner/CSV comparison scripts are written.
Both CPU/CUDA builds compile. Options bound N to2/4/8/16, orders1–12, fixed crossing
and chain topologies, explicit sampled-index counts, methods and repeated timings.
Raw CSV includes exact selected indices, result norms/errors, setup and per-index
costs. Runner saves source/binary hashes, prefixes/libraries and raw timing samples.
The production evaluators are untouched.

CPU order1/6/12 smoke passed. GPU order1/3/8 smoke exposed a high-order comparison
outside the initial tight relative gate (1e-10 plus1e-28 absolute). Investigating
precise error before deciding acceptance; do not weaken it blindly. Some order8
contributions are ~1e-23 and relative errors are large despite tiny absolute error.
Diagnostic job99949 rebuilds/runs order8 with scientific error reporting. Check its
status and /tmp/xca-small-diagnostic.csv before restarting. Previous build jobs ended;
preparation build/smoke logs /tmp/xca-small-{cpu,cuda}-build.log and related CSVs.

Supervisor now accepts configured effort (default remains medium); 24 tests passed,
including exact model/effort preservation. Preparation runtime:
cbuild-supervisor-small/preparation/. Existing boot marker now launches its config;
old completed-goal runtime preserved. Goal identity/pause/quota checks unchanged.
At transition, use a fresh exploration runtime/objective, select Sol6.1/medium using
`thread/settings/update` (installed protocol supports it), and verify thread/read.
Do not launch a second inference worker or change permissions. No model change yet.

Next: resolve benchmark numerical coverage; validate runner/compare and CPU/GPU
high-order/method cases, record quick/throughput baselines and a small N4 profile.
Then create plan_dense_cuda_small.md with strategy priorities, isolated experiment
layout, robust numerical gates, measured success/stop criteria and handoff. Complete
preparation goal only after its artifacts are committed; create exploration goal and
hand off to the requested model. Read this checkpoint, AGENTS.md and status on resume.

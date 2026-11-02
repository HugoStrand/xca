# Diagram index API cleanup work log

## Plan checkpoint — 2026-11-02

Starting source: `c4589ac` on `dev_backbone`.
Plan: `plan_diagram_index_api_cleanup.md`.

The agreed boundary is one topology per Python evaluator call, returning the
current rank's contribution, and one prepared Backbone per local C++ numerical
call. Python reduces completed sums. Single-index, vector and interval selections
will no longer be evaluator API; fixture indexing remains test-only.

Two independent read-only reviews checked MPI/paired scheduling and numerical
test migration. The revised plan explicitly preserves production symbolic-input
validation and complete-result public Solver helpers, checks MPI initialization,
and gives dense paired evaluation its own compact enumeration. No production
source changed during planning. No new benchmarks are requested.

Implementation is pending. Keep source and work-log commits separate, use terse
house-style subjects, preserve unrelated user changes, and do not push.

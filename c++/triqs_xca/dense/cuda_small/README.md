# Native CUDA-small evaluator

Native production foundation at7d00486, batching atd341d73 and native ranges at68be63d. Full C++ API is native: ordinary/paired
self-energy, dynamic interactions, SPGF and correlators. Calls are serialized per
instance, with owned nonblocking stream, BLAS/solver handles and upload event handoff.
Other instances do not rely on the shared NDA solver synchronization setting.

Ordinary and paired SE batch rank16 dimensions4/8 using widths8/32/128. Smaller tails use
native serial traversal; each contribution accumulates into the same device output
in caller order. Metadata grows within a workspace budget rather than fixed prototype
limits; three width caches are bounded. Other shapes/methods remain native sequential.
Half-open work-range methods cover ordinary/paired SE, SPGF and one-time correlators,
including paired fb(0)==0 filtering, without materializing a full index list.
Python Solver now supports explicit dense_backend='cuda_small' and submits native
ranges for MPI work. CPU and library CUDA are separate choices; GPU imports are lazy.
Backend and visible-device preferences survive copies/HDF5. Dynamic interaction
channels are included in ordinary/paired batching. Broader performance qualification
and profiling remain in progress.

Rank16 dimensions4/8 provisionally use grouped real-FMA products with tiles of four;
other dimensions/ranks use library products. Vertex products currently use library
operations. Cached pivoted library LU is retained. The small kernel target compiles
power-of-two product families; compiled availability is not numerical qualification.
Production thresholds require GH200-class qualification. Current A4000 evidence is
local only, and extended method/order/rank/seed coverage remains necessary.

CMake `XCA_CUDA_SMALL=AUTO|ON|OFF` controls this backend. AUTO enables it when NDA has
CUDA and a CUDA compiler is found; otherwise it preserves CPU/library-only builds.
ON requires both; OFF disables custom kernels. `XCA_HAVE_CUDA_SMALL` is the public
availability definition. Set `CMAKE_CUDA_ARCHITECTURES` for the deployment hardware
(e.g.86 on the local A4000); the source does not hardcode that GPU architecture.
Installed target exports include the custom kernel library and CUDA runtime dependency.
A staged external C++ consumer and installed Python backend tests pass. AUTO
without a CUDA compiler has an isolated configure check; optional library-only
Python-extension validation remains in progress.

Nine `dense/cuda_small/*` CTest entries reuse existing C++ test sources with the
native class substituted. These cover physics, paired evaluations, dynamic operator
rotations/parity, statistics, layouts and flat-index APIs; no baseline-delegating
adapter is involved. CPU-only static library and CUDA-small static library builds
pass. Library-only CUDA with OFF also builds and runs all four NDA GPU smoke cases.
CPU-only ON rejects the configuration clearly. Full CPU/CUDA suites and staged
Python/C++ consumers pass. Two-rank Python Solver tests pass with both ranks sharing
the local GPU. Selected cache, correlator and lifetime paths pass memory/leak checks;
this is not sanitizer coverage of every path. Broader performance and resource
accounting qualification remains in progress.

Choose the visible GPU before constructing CUDA objects: Solver accepts
`gpu_device=None` (current device, respecting `CUDA_VISIBLE_DEVICES`) or a visible
ordinal. Keep one unchanged device per process. Native evaluation checks the
construction device before GPU work; destruction temporarily selects that device
through member cleanup and restores the caller's selection. Device-switch rejection
and restoration have a two-GPU regression, skipped on the single-GPU local machine.
Independent instances and cross-thread destruction are tested locally; concurrent
calls on the same instance remain unsupported.

Memory reporting separates `device_workspace_bytes()` (owned device array payloads),
`unified_workspace_bytes()` (owned solver status), and
`host_workspace_bound_bytes()` (model snapshots and conservative metadata bounds).
`workspace_bytes()` combines these categories for the resident budget. Host bounds
include callback payloads, pointer-table objects and the maximum size of cppdlr's
six private lazy time-domain caches without materializing them. Schedule-vector
replacement is checked while its old allocation is still live. Small inline callback
captures may be counted twice, deliberately conservatively.

This is an application storage budget. CUDA-library/context storage, allocator
bookkeeping, retained shared mesh data and caller inputs/results are excluded.
Constructor setup and per-call host staging are not a total process-memory cap;
correlator resize preflight separately bounds its endpoint/output replacement peaks.
Nsight captures quantify opaque library allocations and unified migrations.
Before creating a cache, admission checks its conservative storage estimate against
both the owned budget and current free-device memory minus an advisory reserve.
Qualified widths128/32/8 are tried in descending order, then resident serial work.
Solver `dense_device_memory_reserve` defaults to64 MiB and survives copies/HDF5.
Changing the native reserve evicts caches. This is headroom for opaque library
storage and other users, not a memory reservation; concurrent allocations can still
fail. The default is conservative and provisional for GH200-class qualification.

`Solver.solve()` coordinates GPU initialization failures before entering MPI
iterations. Direct `init_diagram_evaluator()` remains local. Two-rank tests inject an
invalid ordinal on one rank, require coherent failure, then recover and solve with a
valid device. This does not establish multi-device hardware behavior.

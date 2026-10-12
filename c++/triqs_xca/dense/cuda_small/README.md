# Native CUDA-small evaluator

Initial production layer at commit7d00486. Full C++ API is native: ordinary/paired
self-energy, dynamic interactions, SPGF and correlators. Calls are serialized per
instance, with owned nonblocking stream, BLAS/solver handles and upload event handoff.
Other instances do not rely on the shared NDA solver synchronization setting.

This first layer traverses indices sequentially. Cross-index batching and bounded
native work submission are still to be integrated; Python does not yet select this
backend. Passing C++ functionality tests does not complete the implementation plan.

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
Installed consumers and compiler-unavailable library builds still need validation.

Eight `dense/cuda_small/*` CTest entries reuse existing C++ test sources with the
native class substituted. These cover physics, paired evaluations, dynamic operator
rotations/parity, statistics, layouts and flat-index APIs; no baseline-delegating
adapter is involved. CPU-only static library and CUDA-small static library builds
pass. Full CPU/CUDA suites, sanitizers, resource policies, Python/MPI integration,
installed builds and full-API performance acceptance remain outstanding.

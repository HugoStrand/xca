# Dense CUDA-small completion audit

Target: GH200-class accelerators. RTX A4000 measurements are local controls.
The goal permits completing achievable local work and documenting unavailable
hardware qualification. This audit does not redefine missing target measurements
as successful qualification.

|Requirement|Current evidence|Status|
|---|---|---|
|Native target and full constructors/API|`c++/triqs_xca/dense/cuda_small/diagram_evaluator.hpp/.cpp`; real/complex atom_diag and beta/eps/itops/FSet; nine reused production CTest entries|Locally verified|
|Ordinary/paired SE, dynamic channels, SPGF/correlators|Full applicable C++ suites; native-rank eight-suite run plus sixteen complex-model pole/channel fixtures; Python physical methods/solve/dynamic tests|Locally verified|
|Power-of-two specialization and fallback|Fixed product/vertex families2/4/8/16/32/64; only measuredN4/N8 ranks16/26/36 admitted; N2 library/per-lane LU control, N16/32/64 library physical screens|Locally verified sampled cases; target qualification unavailable|
|Strict numerical acceptance/rejections|Shared harness finite checks and unchanged1e-280+1e-10 norm gate; individual and aggregate comparisons; rejected fused LU/N2 grouped-solve/N16 CPU-disagreement controls retained|Locally verified; no universal equivalence claim|
|Repeat/crossover evidence|Native rank/fused/power_two results, three-process two-seed cohorts, matched retained CUDA/CPU larger-dimension screens; cold/setup/context costs and raw manifests|Local staged evidence; target crossover unavailable|
|Bounded work and metadata|Native half-open ranges,128 descriptors, widths8/32/128, three caches, checked arithmetic, dynamic preparation bounds;8192 enumeration and long/range tests|Locally verified|
|Deterministic sums, tails, changed inputs|Same device output, caller-order contributions; widths/tails129/259, duplicates, changed G/topology/method/endpoint tests|Locally verified|
|Allocation/resize/retry safety|Separate prepare/execute; injected submitted errors propagate without replay; budget/cache recovery, category tests and correlator replacement preflight|Locally verified; opaque library/allocator/process memory excluded explicitly|
|Owned streams/handles/status/handoff|Private BLAS/solver/stream, persistent distinct status slots with safe draining; delayed pinned default-upload production test|Locally verified|
|Independent-instance contract|Actual simultaneous ordinary/paired computation on two objects; interleaving and cross-thread destruction; sanitizer evidence|Locally verified; same-instance calls serialized|
|Device placement/lifetime|Solver visible ordinal/current default, process pinning, native construction-device checks and cleanup; two-rank single-GPU test|Locally verified; two-visible-device regression skipped|
|MPI orchestration/errors|Native range submissions, paired filtering, empty/uneven partitions; collective eval/setup failure and recovery; two real ranks each pass12 backend tests|Locally verified sharing one GPU; multi-device hardware unavailable|
|Backend persistence/unavailability|CPU default, explicit cuda/cuda_small, old archive defaults, copies/HDF5, lazy optional imports, unavailable backend errors; default None repair|Locally verified|
|CPU/library/small build options|AUTO/ON/OFF; CPU and library-only without CUDA language/compiler; compiler-missing AUTO fixture; architecture configurable, PIC/export dependencies|Locally verified|
|Generated bindings ON/OFF|Optional alias modules; latest regeneration matches checked-in wrappers; normal builds OFF; CPU availability unchanged|Locally verified|
|Full current suites|Latest CPU67/67; CUDA75/75 including delayed upload; library-only67/67; MPI and installed scripts|Locally verified retained logs|
|Installed use|Normal prefix CUDA-small Python12 tests; fresh external C++ consumer; library-only staged Python12 tests with3 Small skips|Locally verified; installed metadata refreshed to1718ca1|
|Memory/profiling/build size|Separate owned device/unified/host bound; before/after timed Nsight tables; selected memcheck/leak checks; thin compile/resource/object-size record|Locally verified sampled paths; no blanket leak/allocation-free claim|
|Real Solver workload comparison|Ordinary and fitted density/spin self-consistency plus SPGF/density observable; CPU/retained/small parity and complete timing|Locally verified; CPU wins these short workloads|
|Preserve baseline/experiments/commit policy|23-file baseline verifier; all experiment folders retained; terse scoped code/data commits separate from plans/logs ondev_cuda|Locally verified; no push|
|Persistent supervisor|Live process with implementation config, Sol6.1 medium standard tier;28 isolated quota/process/transport tests; observed wait/active transitions|Locally verified; no forced account quota changes|
|Bounded current high orders10/12|Four current admitted-rank/dimension screens, crossing/chain ordinary/paired8-index batches, all exit0|Locally verified|
|Fair same-policy library-product control|Updated experiment keeps pointwise fusion identical; four controls each pass24 groups; matched product-family report retained|Locally verified|
|Final source/package provenance and audit closure|Installed metadata refreshed; current integration/build/workload/control artifacts and final checkpoint committed separately|Locally verified|

## Capability and dispatch table

All methods are native; no cuda_small path delegates to dense/cuda. The retained
library evaluator is separately selectable as `dense_backend='cuda'`.

|Dimension/rank/environment|Ordinary/paired SE|Dynamic channels|SPGF/one-time correlators|
|---|---|---|---|
|N4/N8,r16/26/36, measured local tuple|Tile4 products and bounded widths8/32/128; smaller tails serial|Same batch traversal, density/spin/rotation tested|Native sequential traversal, same eligible product family|
|N2 or other power-of-two/rank|Native sequential library products; safe N2 RHS shape|Native library route|Native library route|
|Unknown hardware/software tuple|Native sequential library route|Native library route|Native library route|

Local tuple: RTX A4000 CC8.6, compiled/runtime13040, driver API13000,
cuBLAS13.8.0,cuSOLVER12.3.4. Library routes remain available; compilation of a
kernel family does not imply its admission. Inputs/cancellation matter, so table
entries summarize measured evidence rather than all-input bitwise guarantees.

Application budgets cover owned payloads and conservative host metadata/snapshots.
CUDA-library/context allocations, allocator bookkeeping, shared mesh storage and
caller inputs/results are excluded. Free-device headroom is advisory. Constructor
and host staging are not a cap on complete process memory. Correlator preflight
checks its own replacement/temporary peaks. These scope limits are part of the
resource policy, not missing hardware capacity assurances.

GH200 and actual multi-device experiments cannot be run on this single A4000.
Before target optimization admission, repeat numerical/physical/solver qualification,
matched crossover and resource measurements there. The block-sparse follow-up remains
separate; variable/rectangular dimensions are not dense acceptance requirements.


## Audit conclusion — 2026-10-13

The achievable local implementation and verification objective is complete. All
required methods are native, explicit backend/MPI/persistence integration operates,
three build variants and installed consumers pass, numerical/physical/resource
contracts are tested, and measured dispatch decisions/rejections are retained. No
local implementation item above remains pending. Same-policy library-product controls
close the final comparison gap. Baseline23 files and all experiments are preserved;
user files remain untouched. Code/evidence and plan/checkpoint commits are separate.

GH200 and true multi-device qualification remain explicitly unavailable rather than
passed. Optimized admission does not extend to those environments. This follows the
objective's instruction to complete achievable local work and document unavailable
target qualification; it does not establish a high-end performance cutoff. CPU
remains Solver default, and no automatic backend choice is enabled.

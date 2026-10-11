# E3 results: bounded cross-index batching

The N4/r16 prototype passes strict CUDA-reference parity and gives useful throughput
for full batches. Three fresh B128 processes repeat the gain. Production is unchanged;
this is ordinary SE through order12, complex double, cached pivoted LU and sequential
caller-order accumulation. No precision/tolerance change or production promotion.

| Crossing workload | Frozen CUDA | CPU in E3 sweep | E3 warm API, three processes |
|---|---:|---:|---:|
| Order6,128 indices |599.061ms |95.245ms |32.727 /32.741 /32.787ms |
| Order8,128 indices |831.037ms |129.747ms |44.459 /44.472 /44.500ms |

These are roughly18.3–18.6× frozen CUDA and2.9× current CPU at full width, with
matching dense-small-v2 topology/index lists, rank/seed/precision. Per-index API time
is256us/348us. Five measured samples per process; raw samples, MAD and provenance
are retained. The source revision is3d7c826 (checkpoint4e83534). Earlier uncached
pre-handoff probes were preliminary and are not used as acceptance evidence.

The capacity sweep demonstrates padding/chunk costs (order8, crossing, median ms):

| Capacity |1 index |8 indices |32 indices |128 indices |Known owned arrays |
|---|---:|---:|---:|---:|---:|
|1 |4.201 |41.696 |166.297 |663.945 |221248B |
|8 |6.500 |6.631 |31.625 |127.973 |1511488B |
|32 |13.492 |13.627 |13.807 |63.995 |5935168B |
|128 |43.338 |43.639 |43.767 |44.531 |23629888B |

One cached schedule only: repeated multi-chunk calls alternate keys and prepare
metadata on every chunk. Thus B1 for128 indices is slower than E1; it is a useful
control, not a recommended selection. Padding128 lanes for one index is very poor.
Capacity selection should follow the actual chunk size; adaptive tail widths and
multiple schedule ownership remain future experiments, not implemented features.
B128 with129 indices costs69.684/94.788ms (orders6/8), almost double128 because the
single-lane tail traverses a full padded chunk and misses the one-chunk cache.
API tests cover B+1 and2B+3 with duplicates, changed inputs and deterministic output.

With cache disabled,128 indices cost36.973/49.913ms (orders6/8), still about2.6× CPU.
This measures complete metadata preparation each call as a changing-index control;
actual index contents are fixed so exact comparisons are retained. Fresh first-call
costs are36.9–37.1/49.8–49.9ms, and constructor setup7.4–7.5ms. Cache reuse saves
about4.2/5.4ms per call; the very first call already beats frozen CUDA and current CPU
including setup. Comparing an uncached call with a cached call of equal first/setup
cost has no extra creation cost to amortize: the second identical call realizes reuse.

Four capacities/throughput profiles, three full-batch repeats, uncached control,
orders through12, rank26, seed31415 and129-index tail timing all pass strict CUDA
per-index/aggregate/changed-G/duplicate/empty comparisons. Sampled errors are zero.
Low-order CPU agreement passes. Known high-order CPU discrepancies are retained:
coverage maximum per-index relative error1.3519 (order10 crossing, absolute3.54e-34); rank26
control2.76e-9; tails0.01013. These are CPU/reference differences, not E3 parity failures.

Five CTest entries pass: ordinary-SE adapters reuse original one-/two-fermion and
complex-rotation tests, plus wide independent-lane primitives and API regressions.
The r16/B128 primitive initially exposed a pageable default-upload race: reconstruction
error0.141 and large coefficients. A persistent default-to-owned-stream event fixes
ordering with no arithmetic change. A deliberately delayed producer fails when the
handoff is omitted, then passes with restored source; both logs are retained. E1/E2
received the same boundary fix and three fresh strict-parity quick repeats each.

N4/order6/B128 Nsight trace:414 kernels,31.852–31.857ms active kernel time, traversal
32.622–32.641ms. GEMMs27.531–27.533ms; solves3.566–3.571ms. One chunk has no inner
allocations/frees or explicit H2D/D2H transfers,6295552D2D bytes,8192UVM bytes (4096
maximum), and74747573B observed peak allocations including reference/library caches.
Known arrays exclude opaque CUDA/library memory and peak is not physical residency.
The graph-independent trace is retained under ignored cbuild-small-profile/batch/;
CSV audit is tracked. Launch geometry resolves the shared CUTLASS complex-double GEMM name:
44 tiny products/vertices use grid(1,1,2048=r*B), totaling26.369ms;40 rank transforms
use grid(8,8,1), totaling1.162ms. Counts match10 convolutions (30 tiny+40 rank) and
14 vertex products. Tiny products are the dominant cost, not the wide rank transform. Solver specialization is lower priority given the current11%
solve fraction. The custom N4/fusion experiment is next.

CUDA memcheck passes all3 API tests and both DLR primitive tests with0 errors;
logs are tracked under results/validation/. Broader rank/shape performance controls
may refine the recommendation. No universal CPU speedup or optimal capacity claim.

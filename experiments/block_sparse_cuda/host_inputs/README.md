# Host-backed propagator experiment

GH200-class accelerators are the target; RTX A4000 measurements screen local
correctness and mechanisms. Production, resident lifetime and row experiments remain.

The private `HostInputDiagramEvaluator` layers host-backed input on the full-shape
lifetime schedule. It borrows the existing host propagator for the synchronous call;
no extra host function snapshot or resident device input array is created. A starting
path/fragment uploads directly into T. Each edge uploads its source into the weighted
value tensor, hands default-stream uploads to the owned stream, applies unchanged
weights/solves/convolution, then releases storage after completed last reads. Preserve
all original host values, including masked correlator roots. Path/summation order,
GEMM dimensions, cutoff and complete outputs are unchanged.

Reference NCA is supported. An explicitly selected RealBatch arena policy rejects
before evaluation because that separate route uploads the full input arena. This is
a disclosed experimental capability restriction, not a silent policy override.
Resident and complete host-sector output placement are supported. Correlator fragments
still reside on device; host inputs alone do not establish full large-correlator fit.

The bound removes one complete input-function payload from the lifetime estimate.
Seven maximum rank tensors include convolution scratch; every admitted entry clears
old scratch before allocating caller storage. Output/fragment, fixed tables, operator/
mapped metadata, trace/result and headroom reserves stay in place. NCA removes only
its inactive rank tensor. Caller-owned host input and complete host output, atom/model
construction and opaque library allocations must be assessed separately. No full N14
physical accuracy/capacity claim follows from metadata formulas.

Build `block_sparse_host_input_tests`, `block_sparse_host_input_benchmark` and
`block_sparse_host_input_cutoff_audit`. Native-disabled CUDA uses the same regular C++
route; CPU omits the experiment. Inherited production and exact baseline controls cover
self-energy/SPGF/correlators, dynamic interactions, high-order ranges, changed G/masks,
output placements, exact budgets/rejection/recovery and tiny-root semantics. Inspect
cutoff audit matrices and decision columns: process completion alone is not acceptance.
Benchmark complete calls with packing/transfers/allocation/drains included. Compare
unchanged reference and resident lifetime on identical fixtures and output placement.

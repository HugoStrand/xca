# Block-sparse CPU workload inventory

`block_sparse_cpu_benchmark` also records bounded complete SE/SPGF/correlator calls
using the unchanged CPU evaluator. Its shared runner/fixtures are in
[block_sparse_common](../block_sparse_common/README.md); it imports no CUDA backend
benchmark headers. Build it with `Build_Benchmarks=ON` and run with the same CLI
workload options as `block_sparse_cuda_benchmark`. CPU-only and CUDA-environment
CPU controls are both supported; record which TRIQS prefix was used.

Build with `Build_Benchmarks=ON`, independently of examples/tests:

```bash
source ~/env/xca-cpu.sh
cmake --build cbuild --target block_sparse_inventory --parallel 6
cbuild/benchmarks/block_sparse_cpu/block_sparse_inventory \
  --modes 6 --partition mixed --dynamic --order 3 --count 32 \
  --shapes /tmp/block-sparse-shapes.csv
```

Partitions and physical fixtures are described in
[block_sparse_common](../block_sparse_common/README.md). The explicit `--topology crossing` (default) and `--topology chain` catalogues
avoid enumerating all pairings. `--first` selects a bounded index
window; `--count` defaults to 32 and is capped at 4096. Counts include rejected block
paths. Setup includes atom_diag, sparse model export and the DLR mesh. Trace time is
mean CPU scheduling time after one warm-up, with `--samples` defaulting to 3 (maximum
7); shape-histogram construction is excluded from the timed scheduling loop.

Actual atom_diag construction is limited to 2–10 modes. Large-case memory estimates
are analytic, with explicit rank and no diagonalization, propagator allocation,
path traversal or timing claim:

```bash
cbuild/benchmarks/block_sparse_cpu/block_sparse_inventory \
  --analytic --modes 14 --partition number --rank 26
```

The summary reports owned model storage, CSR array payload and a conservative owned
storage bound. Neither includes atom_diag or allocator bookkeeping. Memory columns
for one function, the largest rank buffer, legacy scratch and initial correlator
side arrays are capacity diagnostics; other outputs, temporaries and library storage
are additional. Future device admission must account for actual buffer lifetimes.

Use `run_inventory.py --binary <executable> --output <machine-directory>` to record
bounded cases, shape CSVs, failures, source hashes and machine metadata. System-specific
measurements belong in hardware-named subdirectories; generic definitions stay here.

`block_sparse_memory_plan` is a host-only capacity diagnostic for the production
reference evaluator. It constructs binomial number-sector dimensions for ten or
fourteen modes without atom_diag, propagators, CUDA queries or GPU allocations:

```bash
cmake --build cbuild --target block_sparse_memory_plan --parallel 6
cbuild/benchmarks/block_sparse_cpu/block_sparse_memory_plan \
  --modes 14 --rank 64 --device-gib 120 --host-gib 64
```

CSV rows cover Reference/Mapped special policy and resident/host-sector output.
Rank1–256 is a requested capacity parameter, not a physical DLR resolution claim.
Two poles,32MiB sparse reserve and1×1 correlator endpoints are declared. Budgets are
GiB, not decimalGB. Mapped width assumes all annihilation fields share a set; extra
retarded operators can change it. Host output is the full self-energy return payload
for both placements. Caller-owned input, model/atom_diag construction, allocator/
library bookkeeping, context/headroom and temporary host copies remain additional.
A within-payload-budget flag is not available-memory admission or qualification.

The host-only planner shares its generic storage calculation with production and
is tested against live evaluator bounds for both special/output policies. Native
CUDA execution still enforces free memory, headroom and configured host output
limits. Target hardware is GH200-class; local measurements and synthetic dimensions
alone do not qualify a full ten/fourteen-mode solver or determine physical ranks.

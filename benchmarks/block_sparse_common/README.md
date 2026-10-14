# Block-sparse benchmark fixtures

Performance target: GH200-class accelerators. Local RTX A4000 results are development
proxies; dispatch and capacity decisions require target measurements.

`fixture.hpp` defines deterministic physical models with unequal onsite energies,
density-density interactions and, for number/mixed partitions, generic complex
hopping. `occupation` uses automatic scalar blocks and diagonal hybridization;
`number` conserves total particle number with binomial sector dimensions; `mixed`
conserves particle numbers in two hopping-connected orbital groups. Compatible
hybridization coefficients are complex within each current symmetry set. Optional
retarded channels are total density and an orbital-transfer monomial, with diagonal
channel coefficients. The current grouping restrictions are preserved.

Analytic sector inventories need no atomic diagonalization. They report payload
capacity at a requested rank, not physical rank/accuracy qualification. Actual model
inventories use the DLR rank for beta=2, Lambda=40 and epsilon=1e-10. Operator snapshots
use direct Fock CSR export; rank functions remain dense within sectors.

Path counts are structural candidates with an explicit all-live propagator mask.
They include paths whose arithmetic is zero or rejected by the retained CPU's
per-path magnitude cutoff. No numerical evaluation or GPU arithmetic is timed.
Record rejected proposals as well as accepted paths; active-only selection cannot
establish complete Solver throughput.

Shape rows describe matrix geometry. `calls` counts rank batches for vertex products,
three rank batches per convolution, individual rank-r solves with their separate RHS
widths, and reference special-vertex q² right/q left products for a set of width q. `special_mix` is an
entrywise weighted reduction: K is the set width and calls is the number of mu
members, not a GEMM dimension. These are workload descriptors, not measured FLOP rates.

## Complete-call controls

`call_runner.hpp` shares CLI options, physical propagator construction, timers and
CSV output. Backend factories live in their respective CPU/CUDA folders. Supported
observables are `se`, `spgf`, `correlator`; partitions are the same as the inventory.
Use `--topology crossing|chain`, `--order`, `--first` and `--count` for explicit
bounded work without topology enumeration. Actual modes are limited to 2–10,
count to 1–8 and samples to 1–5. N=14 remains an analytic/primitive-proxy tier.

`atom_ms` includes atom_diag, coefficient/propagator preparation and endpoint model
setup. `evaluator_setup_ms` is constructor time (including sparse export, device
context/handles and uploads for CUDA). `call_ms` is host wall time of the complete
synchronous API, including discovery, input upload, device arithmetic and output
transfer; flattening and CSV output are outside it. One untimed call warms caches.
Raw samples are retained rather than reporting only a mean. Repeated results must
match the warm result exactly. CPU is the unchanged evaluator; this is not the
algorithm-matched CPU control required for later DLR experiments.

CUDA first evaluates a rotated-operator CPU control. `cpu_control_ms` includes that
call and flattening, not its constructor, so use the separate CPU frontend's
`call_ms` for latency comparisons. Numerical screening requires finite results and
absolute error <= 1e-10*max(CPU norm, CUDA norm)+1e-14; strict exact-input/high-order
arithmetic qualification is in the test suite. Signal and errors are reported even
for weak/zero windows. CPU rows have zero comparison errors because no independent
control is run there. CUDA path counts are structural candidates; CPU count columns
are `NA`. Do not select only nonzero windows to claim complete-solver speedup.

`host_peak_kib` is Linux process RSS high water, including setup/control. Device
bounds are conservative explicit-storage diagnostics. `device_peak_delta` samples
global free-memory drops relative to call entry, excludes preexisting resident
storage, includes other processes and can be zero with allocator reuse; it is not
an allocation count or continuous peak measurement. Run paired cases in isolation.

Record the bounded campaign sequentially with:

```bash
source ~/env/xca-cuda.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 benchmarks/block_sparse_common/run_calls.py \
  --cpu-binary cbuild-cuda/benchmarks/block_sparse_cpu/block_sparse_cpu_benchmark \
  --cuda-binary cbuild-cuda/benchmarks/block_sparse_cuda/block_sparse_cuda_benchmark \
  --machine thinkstationP620_RTXA4000 --campaign reference-01
```

Raw backend results go under `block_sparse_cpu/<machine>/<campaign>` and
`block_sparse_cuda/<machine>/<campaign>`; combined samples/provenance go here under
`<machine>/<campaign>`. Generic definitions remain at the first directory level.
The recorder stores source/binary/installed-header hashes, commands, compiler,
TRIQS prefix, machine metadata, failures and timeouts (120 s per subprocess default).
It refuses an existing comparison campaign directory. Its small campaign covers
all observables, ordinary/retarded channels, 4/6-mode regimes and bounded 8/10-mode
limits. It does not establish a dispatch winner or target-hardware defaults.


## MPI ownership and phases

Configure with Python support and benchmarks enabled. CMake writes
`<build>/benchmarks/block_sparse_common/mpi_environment.json`, using that build's
Python interpreter, package and dependency sources. Source the matching CPU/CUDA
environment, then run a bounded two-rank case:

```bash
python3 benchmarks/block_sparse_common/run_mpi.py \
  --environment cbuild-cuda/benchmarks/block_sparse_common/mpi_environment.json \
  --launcher-arg=--oversubscribe --ranks 2 --record /tmp/mpi-command.json -- \
  --backend cuda --operator-backend ordered --special-backend mapped \
  --partition number --modes 3 --count 3 --samples 2 --output /tmp/mpi-results
python3 benchmarks/block_sparse_common/summarize_mpi.py /tmp/mpi-results \
  --output /tmp/mpi-summary.csv
```

The output and command files must not already exist. `run_mpi.py` sets one OpenMP/
BLAS thread and bounds the launcher process group (default240seconds). Alternate site
launchers use repeated `--launcher-arg`; a private interpreter alias carrying site
MPI metadata must resolve to the configured Python. The communicator size and a
collective preflight are checked before CUDA setup. Fixtures do not import tests.

Two fixed order-three topologies are crossing `(03)(14)(25)` and chain `(02)(14)(35)`.
`--first`/`--count` select the same bounded window of each; no topology enumeration
or full-order throughput extrapolation. Models use the shared C++ fixture's
Hamiltonian/coefficient formulas, with explicit occupation/total-number/two-group
partitions; optional retarded channels are total density and orbital transfer.
Solver constructs G0 and the evaluator through the public integration. This is a
fixed-input benchmark, not an iterative physical Solver calculation.

Three policies run after one complete warm-up, with1–3 samples per process:

- `topology_split`: current contiguous ownership within each topology, but one API
  call per indexed job and one coalesced canonical payload reduction.
- `flat_cyclic`: round-robin ownership of flattened topology/index jobs, with the
  same individual calls and reduction payload. Every indexed output has one owner;
  exact equality across ownership/sample changes is required. A rank-zero CPU
  per-index control is built outside the measurements.
- `solver_ranges`: contiguous range calls and per-topology/per-block reductions,
  matching Solver's grouping and call pattern. Grouping can change roundoff; the
  full selected output is compared to the same CPU indexed sum with the existing
  complete-call screen. Error coordination uses an allgather, rather than Solver's
  scalar failure allreduce. This is not a byte-for-byte timing of Solver internals.

`phases.csv` records per-rank synchronous API time, host packing, collective error
coordination (including straggler wait), explicit arrival barrier, payload reduction,
final accumulation and total recorded wall time. API time includes discovery,
host-to-device copies, kernels and return copies. It is **not** isolated device time.
Verification/CPU reference work is outside each recorded interval; recorder/job
bookkeeping remains in total wall time. The barrier separates arrival wait from
payload reduction, so reduction timing alone is not communication cost in production.
Packing may return a contiguous view. Payload bytes are each rank's user buffer per
reduction, not measured wire traffic. Phases can overlap across ranks; maxima cannot
be summed to reconstruct critical time. `summary.csv` reports median critical-rank
wall times and maximum/mean local API-time imbalance, preserving per-sample ranges.

`jobs.csv` retains individual job times and zero/weak signals; the summarizer checks
complete unique ownership, rank/sample coverage and finite metrics. Setup/control
time, source/extension hashes, communicator size, topology counts, rank PIDs/hosts
and CPU affinity are retained. Compare CPU and CUDA and test count smaller than rank
count to retain empty-owner/tail controls. These bounded measurements establish no
automatic scheduling or device-dispatch threshold.

For a **separate instrumented** CUDA trial, use `--profile` and prefix each worker
with `nsys profile --trace=cuda --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi --capture-range-end=stop --output=/tmp/mpi-%p`, using
one repeated `--worker-prefix` argument per token. Capture starts after warm-up and
covers all measured policies. Nsight memory/kernel reports then give actual device
copy durations/bytes/counts for each captured process. Map report PIDs to manifest
rank metadata; capture-wide device durations are not policy-specific wall-time
components. Preserve uninstrumented timings independently; profiler overhead and
shared-device contention invalidate direct latency comparisons. Target remains
GH200-class, and local RTX A4000 traces do not qualify target transfer bandwidth.

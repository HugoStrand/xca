# Shared dense benchmark fixtures and tools

`harness.hpp` defines the deterministic physical fixture, topology/index selection,
result checks and timing/CSV contract. `runner.py` records raw samples, provenance
and summary statistics; backend directories provide executable defaults.
`compare.py` compares matching summaries. `solver_workload.py` measures complete
Solver workloads across available backends.

The fixture uses a power-of-two occupation basis with fermionic sign conventions,
two hybridization poles and complex coefficients. Its propagator is a dense Fourier
rotation of a normalized diagonal imaginary-time propagator. Default inputs are
N=4, beta=2, DLR cutoff=10, tolerance=1e-12, orders 1/3/6/8, crossing/chain topologies,
and one/eight selected indices. Supported dimensions are 2/4/8/16/32/64; the CLI
bounds order at 12 and selected indices at 4096. Larger orders need small samples
rather than exhaustive topology or index enumeration.

Index selection is seeded and stratified across the full long-index range.
Equivalent crossing/chain topologies at orders one/two are measured once. Counts
are clamped to available indices, and exact index lists are recorded. Paired counts
refer to seeds covering both directions, so they are different work units from
ordinary self-energy counts.

## Timing and correctness

Each measured evaluator is freshly constructed after untimed correctness checks.
Construction and CUDA-context initialization are reported separately. Sample -1 is
the first full-vector call on the fresh evaluator; the process/context is already
warm. Untimed warmups precede repeated measurements, which alternate CPU/candidate
order. Wall times cover the host-facing API, transfers, output construction and
completion synchronization. The summary reports median, min, max and median absolute
deviation (MAD). Use at least three fresh processes before interpreting small gains.

The cuBLAS evaluator supplies the GPU reference. Individual indices and aggregate
results must meet max absolute error <= 1e-280 + 1e-10 times the reference max norm.
The tiny floor rejects zero in place of small high-order contributions. Non-finite
values or shape mismatches fail. Duplicate/empty index sets and changed propagator
data are checked before timing. CPU agreement is additionally required through
order three; higher-order CPU differences remain reported diagnostics because
conditioning and cancellation can magnify relative errors. Analytic evaluator tests
remain necessary to assess physical accuracy.

`owned_bytes` describes candidate-owned workspace while a separate reference is
alive; it is neither peak process memory nor physical device residency. First-call
costs, reference norms and numerical errors must accompany speedup claims.

## Recording and comparison

Backend runners support `quick`, `throughput` and `coverage` profiles. Arguments
after `--` override profile defaults. Output directories must be new and contain
`samples.csv`, `summary.csv`, `manifest.json` and `stderr.log`. Failed execution does
not produce a successful summary. Manifests identify commands, revision, source and
binary hashes, machine, library resolution and environment. The runner forces one
OpenMP/OpenBLAS thread; run serially on an otherwise idle GPU.

```bash
python3 benchmarks/dense_common/compare.py reference/summary.csv candidate/summary.csv \
  --reference-backend cpu --candidate-backend cuda_small_b128
python3 benchmarks/dense_common/solver_workload.py \
  --output results/my_machine/solver-workload/run-01 --samples 3
```

Comparisons require matching fixture keys, including exact indices and seeds.
Speedup is reference median / candidate median; both MADs are reported. Review
manifests before comparing hardware, compiler/library environments or revisions.
The Solver control includes fitting/setup, self-consistency, SPGF and a density
observable; use the intended Python package with its matching runtime environment.

## Backend adapters

CPU-only entry points call `run_workload` directly. GPU entry points supply a
`Candidate{name, evaluate, bytes}` factory, a reference factory and `Runtime` hooks
for initialization, completion and scoped profiling. The cuBLAS adapter lives in
`../dense_cuda/adapter.hpp`; the native-small factory lives in `../dense_cuda_small/`.
An adapter must return an owning host tensor, use current fixture inputs and reject
unsupported methods explicitly. Lazy preparation is charged to the first call;
completed results must not be cached.

Store machine-specific reports in hardware subfolders. [Workstation reports](thinkstationP620_RTXA4000/README.md)
cover RTX A4000; the optimization target is GH200-class hardware and needs separate
measurements.

# Small-matrix dense evaluator benchmark

[Recorded baseline and profiling results](RESULTS.md).

A bounded C++ benchmark for optimization experiments, with the production CPU and
CUDA evaluators as references. Default: N=4, rank16, orders1/3/6/8, crossing/chain
topologies and one/eight selected indices. No combinatorial topology enumeration.
The original larger-matrix benchmark remains in `benchmarks/dense_cuda/`.

From the repository root:

```bash
source ~/env/xca-cuda.sh
(cd cbuild-cuda && bash ../cmake-cuda.sh)
cmake --build cbuild-cuda --target dense_cuda_small_benchmark --parallel 12
python3 benchmarks/dense_cuda_small/run.py --output cbuild-small-runs/quick-01
```

The output directory must be new. It contains raw `samples.csv`, `summary.csv`
(median/min/max/MAD), `manifest.json` (command, binary/source hashes, revision,
prefixes, loaded libraries, GPU and thread settings), and `stderr.log`.
A failed process/comparison does not produce a successful summary. The runner forces
one OpenMP/OpenBLAS thread. Run benchmarks serially on an otherwise idle GPU.

```bash
# Amortization versus index count; about a minute on ts, depending on the case set.
python3 benchmarks/dense_cuda_small/run.py --profile throughput --output cbuild-small-runs/batches-01
# Bounded higher-order coverage, all three compute families.
python3 benchmarks/dense_cuda_small/run.py --profile coverage --output cbuild-small-runs/coverage-01
# Custom narrow run; arguments after -- override profile defaults.
python3 benchmarks/dense_cuda_small/run.py --output cbuild-small-runs/order8-01 -- \
  --orders 8 --topologies crossing --indices 1,8,32 --samples 9 --lambda 40
# CPU-only build/prefix control.
source ~/env/xca-cpu.sh
cmake --build cbuild --target dense_cuda_small_benchmark --parallel 12
python3 benchmarks/dense_cuda_small/run.py --binary cbuild/examples/dense_cuda_small_benchmark \
  --output cbuild-small-runs/cpu-01 -- --backend cpu
```

`--help` lists supported bounds: N2/4/8/16, orders1–12, up to4096 indices per case.
Use small subsets at high order. A deterministic seeded, stratified sample selects
indices across the full long-index range; exact index lists are recorded. Crossing
pairs are `(i,i+m)`; chain pairs are `(0,2),(1,4),...,(2m-3,2m-1)`. Equivalent
order1/2 topologies are timed once. Counts are clamped to the available indices.
Paired counts refer to seeds, each covering two directions; they are not comparable
work units with ordinary self-energy indices. Samples of different counts generally
select different indices. Compare candidates on exactly matching cases/index lists.

Each case records first full-vector timing (sample -1), performs untimed warmups,
then alternates CPU/candidate execution order across repeated measurements. Setup
and CUDA context initialization are separate. Each timed evaluator is freshly constructed after untimed correctness checks, so
first-call graph/schedule preparation is charged. The process/context is already warm. Wall time includes the complete host-facing call,
uploads/downloads, final synchronization and normalization to an owning NDA tensor.
`owned_bytes` covers the candidate; the separate reference evaluator also remains
alive. This is not peak process/device memory. Nsight supplies kernel/copy evidence;
CSV wall times are the primary performance metric. Tiny cases should be rerun in
at least three fresh processes before calling a small change a speedup.

## Correctness and numerical limits

Candidates are checked against the **unchanged CUDA evaluator** for every selected
index and every timed vector result. Gate: max absolute error <=1e-280 +1e-10 times
reference max norm. The tiny floor prevents accepting zero in place of small
high-order contributions. The harness also checks duplicate indices, empty subsets,
and a changed propagator at the same host address before timing. Non-finite values
or shape mismatches always fail. Existing analytic/unit tests remain mandatory for
experiment acceptance; the benchmark does not replace them.

CPU results are calculated for every case and individual selected index. Orders1–3
must also meet the CPU comparison gate. Higher orders report CPU absolute/relative
errors without using them to redefine CUDA-baseline parity. Initial probing found
an order8 contribution with norm1.18e-18 and CPU/CUDA difference4.85e-25 (4.10e-7
relative); an order12 SPGF contribution near2.65e-41 differs by about0.25% relatively.
Repeated DLR transforms and cancellation make relative errors significant at these
scales. A speedup therefore means preserving current CUDA results, not establishing
new high-order physical accuracy. Record CPU discrepancy and reference norm alongside
speed; investigate numerical behavior before production use at such orders.

## Comparing and adding experiments

```bash
python3 benchmarks/dense_cuda_small/compare.py \
  cbuild-small-runs/quick-01/summary.csv cbuild-small-runs/quick-02/summary.csv
python3 experiments/dense_cuda_small/baseline/verify.py
```

Comparisons require identical case sets and selected indices. Speedup is reference
median divided by candidate median; both MADs are shown. Use `--reference-backend cpu`
or `--candidate-backend EXPERIMENT_NAME` as appropriate. Do not compare prefixes or
hardware silently; review the manifests. The previous large-matrix report documents
a CPU-prefix timing discrepancy, so use matched environments for optimization claims.

Each experiment gets its own folder under `experiments/dense_cuda_small/`, with an
independent target, code, README and raw/summary results. Its main includes this
header and calls `dense_cuda_bench::run(argc,argv,factory)`. The factory accepts
`Fixture&` and returns `Candidate{name,evaluate,bytes}`. `evaluate(method,topology,indices)`
must return the same owning host tensor and observe the current fixture propagator;
`bytes()` reports owned device buffers. Unsupported methods must fail explicitly.
Measure setup in the factory; work lazily done on first call remains charged to that
call. Never cache results. Cache topology schedules/graphs only with valid input
invalidation and report preparation costs separately when added.

The source manifest pins 23 production files to baseline064b64e. Leave those files
unchanged; experiment code may use its own namespace and adapters. Shared harness
changes must preserve fixture/measurement semantics or increment the fixture version
and establish a new baseline. No automatic promotion of experimental code.

# Dense CPU batching

This benchmark compares the dense scalar evaluator using fast convolution with
`dense::BatchedDiagramEvaluator`. It uses two fermion modes, so every backbone
matrix has Hilbert-space dimension **4**. All timings are CPU timings.

Build and run from a configured CPU build tree:

```bash
source ~/env/xca-cpu.sh
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF
cmake --build cbuild --target dense_cpu_batched --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/dense_cpu_batched/dense_cpu_batched > timings.csv
```

The executable accepts an optional case-name substring, for example `r103`,
`order5`, `correlator`, or `topology/self_energy`. Run it directly, without an MPI
launcher. Avoid other builds or benchmarks while measuring. Machine-specific
results and environment details belong in hardware-named subdirectories.

## Workloads

The model has complex hopping, a density-density interaction, two hybridization
channels and two retarded density operators with complex off-diagonal couplings.
Its propagator contains complex, noncommuting matrix values. The fixture uses the
same construction as the dense batching correctness tests.

Two DLR meshes are tested at beta 2: the small mesh from the evaluator test
fixture, and a mesh with dimensionless cutoff 100000 and tolerance 1e-12. The CSV
records their actual ranks. The larger rank is approximately 100.

For each mesh, prepared-backbone timings cover ordinary self-energy, paired
self-energy, and a fermionic correlator at orders 2, 3 and 5. Each order uses the
fully crossing topology `(a, a + order)`. The benchmark samples at most **63**
backbones from the relevant index space, using stratified sampling and seed 1789.
At order 2 it uses the entire smaller index space. Paired samples come from the
compact pair index space; one sample includes both distinguished-arc directions.
The fixed bound avoids an exponentially expensive complete high-order sum.

Prepared timings exclude model/Action construction, evaluator construction and
backbone preparation. They include the numerical evaluations, returned-result
allocation and accumulation into the final local array. Every case uses the same
backbones and propagator across variants. The backbones share order and
distinguished-vertex position, making them eligible for one batching cohort.
Mixed topology/order workloads may produce shorter cohorts.

Two additional cases evaluate a complete order-2 ordinary self-energy topology,
including backbone preparation and the block-Green-function result wrapper.
These are labelled `topology`; their timings must not be mixed with `prepared`.

## Comparison and validation

The variants are the scalar fast evaluator and batch capacities 1, 2, 4, 8, 16,
32 and 64. The 63-backbone samples exercise partial final batches; larger
capacities have no extra work added to fill them. Workspace allocation is outside
the timed region.

Batch capacity 1 is an important control: the batched implementation also changes
vertex workspace reuse and the number of returned-array allocations compared
with individually calling and summing scalar evaluations. Speedup over the scalar
baseline therefore measures the complete implementation change. Speedup over
batch 1 more directly measures grouping multiple backbones into shared DLR
transforms. These ratios do not isolate convolution alone.

Every output is checked against the scalar fast result before and after timing,
requiring finite values, a nonzero reference and maximum absolute discrepancy no
greater than 2e-10 times the largest reference magnitude. This checks both the
first call and reuse of temporary storage. Independent analytic convolution and
full evaluator correctness tests remain separate from the benchmark.

Timing uses five trials and rotates variant order between trials. Each trial
repeats a call one to three times, calibrated toward 5 ms. CSV columns contain
median/minimum/maximum microseconds **per call**, repeat/trial counts, error,
output magnitude, speedup over scalar fast, and speedup over batch capacity 1.
A prepared call sums all `backbones` entries; compare per-backbone costs only
within the same evaluation kind. Pair entries represent two directions.

The benchmark does not measure an impurity-solver iteration, MPI scaling, or GPU
performance. A speedup here should be followed by a representative solver-level
measurement before changing the default evaluator.

The [small-N CPU experiments](experiments/small_n/README.md) isolate convolution
stages, fixed-size products, real DLR arithmetic and repeated-edge reuse without
changing the production evaluator.

The [few-orbital CPU benchmark](../dense_cpu_few_orb/README.md) compares the
subsequently improved scalar evaluator with handwritten products and edge operand
reuse. Its results use the same shared physical fixture and separate source
provenance; they should not be mixed with older scalar baseline numbers.

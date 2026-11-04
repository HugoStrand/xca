# Dense CPU evaluation for few orbitals

Compare the original `dense::DiagramEvaluator` (including reusable output buffers)
with `dense::DiagramEvaluatorFewOrb` at Hilbert-space dimension **N=4**, i.e. two
fermion modes. The new evaluator uses handwritten 4×4 complex products, real DLR
transforms/solves, and optional reuse of edge operands. Other dimensions use BLAS
for matrix products; their performance is not established by this benchmark.
Both implementations remain available; FewOrb is currently a C++ API.

## Build and run

From a configured CPU build tree:

```bash
source ~/env/xca-cpu.sh
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF
cmake --build cbuild --target dense_cpu_few_orb --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/dense_cpu_few_orb/dense_cpu_few_orb > timings.csv
```

An optional case substring selects workloads, e.g. `r103`, `order5`,
`topology/self_energy`, or `prepared/correlator`. Run directly without MPI and
without competing builds/benchmarks. Store results in hardware-named subfolders.

## What is measured

The [shared fixture](../dense_common/n4_fixture.hpp) is also used by the dense
batching benchmark. It includes complex hopping, hybridization, two retarded
density operators, and a deterministic dressed propagator. Two DLR meshes at
beta 2 give ranks 13 and 103 on the recorded installation. The higher rank uses
dimensionless cutoff 100000 and tolerance 1e-12.

Every case compares three variants:

- `dense`: the original evaluator with direct GEMM into reusable buffers.
- `few_orb_uncached`: handwritten N=4 products and real DLR arithmetic, retaining
  no edge operands between convolutions.
- `few_orb`: the same arithmetic plus lazy edge operand reuse, with the default
  8 MiB numerical cache budget.

The uncached implementation also changes scratch handling; its comparison with
dense does not isolate real DLR arithmetic from fixed-size products or buffer
management. The separate kernel experiment measures only the small products.

For a recipe q containing the ordered edge kernel factors, the cache stores
`C_q = vals2coefs(G * kernel_product_q)` and `H * C_q`, each shaped `(r,N,N)`.
The key preserves factor order and multiplicity. An empty recipe represents G.
No full convolution matrix is stored. Left and right traversals share the same
prepared edge operand; matrix multiplication order still follows the backbone.

Cache validity resets at each public call, including when G changes in place.
Existing allocations are reused. Timings include filling every required cache
entry on each call; they are **not** repeated evaluations with already valid
numerical operands. The budget limits the two cached tensor payloads, excluding
keys/container metadata and fixed scratch. When full, existing entries remain
usable and other recipes are evaluated without retention. The CSV reports hits,
misses, entries and active payload bytes for the final call. Retained allocations
may reflect a larger preceding call.

Prepared cases sum at most 63 stratified backbones with seed 1789, covering
self-energy, paired self-energy and fermionic correlators at orders 2, 3 and 5.
Every case uses the fully crossing topology `(a,a+order)`. Model, Action,
evaluator and backbone construction are outside timing. Scalar prepared calls
return one array per backbone and the driver sums them; the FewOrb span overload
returns one sum. Consequently the dense-to-uncached ratio includes this API and
allocation difference as well as arithmetic changes.

Complete topology cases cover all three evaluation kinds at orders 2 and 3.
These include backbone preparation and use matching public-call boundaries and
result wrappers. **Use these rows for the primary evaluator comparison.** The
`few_orb_uncached` time divided by `few_orb` time isolates edge reuse within
either workload category.
Order-five results remain bounded samples, not full topology sums. Paired
backbones represent both distinguished-arc directions.

Five timing trials rotate variant order. Calls repeat one to five times, with
calibration toward 5 ms per trial. CSV times are microseconds per complete call,
with minimum/median/maximum, errors and speedup over the updated original.
Allocations are warmed before timing; evaluator construction and first-use
allocation cost are not measured. Outputs are checked before and after timing
for finite values, nonzero reference magnitude, and relative maximum error
below 2e-10. These are evaluator timings, not full impurity-solver iterations.

## Optional Eigen comparison

Eigen is used only by a separate kernel experiment. Provide an existing Eigen
header directory when configuring:

```bash
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF \
  -DXCA_BENCHMARK_EIGEN_INCLUDE_DIR=/path/to/eigen
cmake --build cbuild --target dense_few_orb_eigen_gemm --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/dense_cpu_few_orb/dense_few_orb_eigen_gemm > eigen_gemm.csv
```

The experiment compares handwritten `small_gemm<4>`, fixed-size row-major Eigen
maps with `noalias()`, and NDA/OpenBLAS GEMM on 103 independent complex matrices.
It tests overwrite and accumulation, both compile-time unit scaling (as used in
the evaluator) and general complex scaling. It checks against a long-double
reference, then measures five rotated trials, normally 1024 stack repetitions.
An optional positive integer overrides repetitions. A compiler barrier prevents
removal of overwritten results. CSV times are nanoseconds per 4×4 product.

This comparison does not replace the full evaluator with Eigen and does not
predict its end-to-end gain. Record compiler architecture flags, Eigen version
and BLAS implementation with each result; fixed-size template code and a BLAS
library's runtime CPU dispatch need not use the same instruction set.

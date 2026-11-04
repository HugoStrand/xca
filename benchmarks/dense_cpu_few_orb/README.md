# Dense CPU evaluation for few orbitals

Compare the original `dense::DiagramEvaluator` (including reusable output buffers)
with `dense::DiagramEvaluatorFewOrb` at Hilbert-space dimensions **N=2,4,8,16**,
i.e. one to four fermion modes. Both use the shared `cppdlr::fast_convolve`
real DLR transforms/solves and reusable convolution storage. FewOrb additionally
uses fixed-size Eigen complex products at N=2 and N=4, and optionally reuses edge
operands. Its other dimensions, including N=8 and N=16, use BLAS for matrix
products. Both evaluators remain available; FewOrb is currently a C++ API.

`small_gemm<N>` provides the Eigen implementation, also available at N=8 and
N=16 for isolated comparisons. The evaluator's dimension dispatch chooses BLAS
at those dimensions following the recorded kernel crossover. The previous
handwritten loops remain available as `small_gemm_handwritten<N>` for comparison.
Eigen 5 is a normal header-only dependency of XCA, resolved by CMake; the source
dependency is pinned to 5.0.1. No benchmark-specific Eigen include path is required.

## Build and run

From a configured CPU build tree:

```bash
source ~/env/xca-cpu.sh
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF
cmake --build cbuild --target dense_cpu_few_orb --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/dense_cpu_few_orb/dense_cpu_few_orb > timings.csv
```

An optional case substring selects workloads, e.g. `N4/`, `N16/r103`, `order5`,
`topology/self_energy`, or `prepared/correlator`. Complete order-three topologies
at N=8 and N=16 require `--full-order3`; they can be substantially more expensive
than the bounded samples. For example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/dense_cpu_few_orb/dense_cpu_few_orb \
  N8/r103/topology/self_energy/order3 --full-order3 > topology_N8_order3.csv
```

Run directly without MPI and without competing builds/benchmarks. Store results
in hardware-named subfolders.

## What is measured

The [shared fixture](../dense_common/n4_fixture.hpp) preserves the original N=4
model and sampling used by the dense batching benchmark. A model of dimension
`N=2^m` has m hybridizing fermion modes and m retarded density operators,
together with a deterministic complex dressed propagator. Models with two
or more modes also have complex nearest-neighbour hopping, static density
interactions, and off-diagonal retarded coefficients. The N=4 Hamiltonian,
coefficients and propagator are unchanged from the original benchmark.

Two DLR meshes at beta 2 give ranks 13 and 103 on the recorded installation. The
higher rank uses dimensionless cutoff 100000 and tolerance 1e-12. Changing N
also changes the number of arc flavours and the complete topology's work count;
the CSV reports the actual number of backbones. Compare variants within a case
rather than interpreting timings across N as pure matrix-kernel scaling.

Every case compares three variants:

- `dense`: the original evaluator with direct BLAS GEMM into reusable buffers
  and the shared real DLR convolution.
- `few_orb_uncached`: FewOrb with Eigen products at N=2/4 and BLAS products at
  other dimensions, retaining no edge operands between convolutions.
- `few_orb`: the same FewOrb arithmetic plus lazy edge operand reuse, with the
  default 8 MiB numerical cache budget.

Real DLR arithmetic is common to all three variants. The uncached comparison
with dense measures the remaining FewOrb implementation changes, including
matrix-product selection and traversal/workspace scheduling; it is not a strict
matrix-kernel-only A/B. At N=8 and N=16 both use BLAS products, making residual
differences visible without an Eigen-versus-BLAS change. The separate kernel
experiment isolates the matrix products. Comparing the two FewOrb variants
isolates edge reuse with all other choices fixed.

For a recipe q containing the edge kernel factors, the cache stores
`C_q = vals2coefs(G * kernel_product_q)` and `H * C_q`, each shaped `(r,N,N)`.
The key sorts factors by `(pole_index, argument_sign)`, preserving multiplicity:
`A*B` and `B*A` share an entry, while `A*A*B` remains distinct. A hash table
maps full keys to cache-entry indices; equality resolves hash collisions.
An empty recipe represents G. The product is evaluated in this same canonical
order, which can change floating-point rounding relative to the baseline.
No full convolution matrix is stored. Left and right traversals share the same
prepared edge operand; matrix multiplication order still follows the backbone.

Cache validity resets at each public call, including when G changes in place.
Numerical allocations and hash buckets are reused; table nodes and keys are
rebuilt on each call. Timings include filling every required cache
entry on each call; they are **not** repeated evaluations with already valid
numerical operands. The budget limits the two cached tensor payloads, excluding
keys/container metadata and fixed scratch. When full, existing entries remain
usable and other recipes are evaluated without retention. The CSV reports hits,
misses, entries and active payload bytes for the final call. Retained allocations
may reflect a larger preceding call.

The numerical cache cost per recipe is `2*r*N*N*sizeof(complex<double>)`, so
doubling N quadruples storage. At rank 103 and N=16 one recipe occupies about
0.805 MiB, allowing nine recipes within the default 8 MiB budget. Higher-order
cases can therefore encounter cache pressure even when the smaller dimensions
retain all their encountered recipes. The fixed budget is deliberate and is
identical across dimensions; consult the recorded hit/miss counts.

Prepared cases sum at most 63 stratified backbones with seed 1789, covering
self-energy and fermionic correlators at orders 2, 3 and 5. Self-energy
samples use the compact pair index space; each backbone sums both directions of
the special arc connected to zero.
Every case uses the fully crossing topology `(a,a+order)`. Model, Action,
evaluator and backbone construction are outside timing. Scalar prepared calls
return one array per backbone and the driver sums them; the FewOrb span overload
returns one sum. Consequently the dense-to-uncached ratio includes this API and
allocation difference as well as arithmetic changes.

Complete topology cases cover both evaluation kinds at order 2 for every
N, and at order 3 for N=2 and N=4 by default. `--full-order3` additionally enables
the N=8 and N=16 cases. These include backbone preparation and use matching
public-call boundaries and result wrappers. **Use these rows for the primary
evaluator comparison.** The `few_orb_uncached` time divided by `few_orb` time
isolates edge reuse within either workload category. Order-five results remain
bounded samples, not full topology sums.

Five timing trials rotate variant order. Calls repeat one to five times, with
calibration toward 5 ms per trial. CSV times are microseconds per complete call,
with minimum/median/maximum, errors and speedup over the updated original.
Allocations are warmed before timing; evaluator construction and first-use
allocation cost are not measured. Outputs are checked before and after timing
for finite values, nonzero reference magnitude, and relative maximum error
below 2e-10. These are evaluator timings, not full impurity-solver iterations.

The current `self_energy` cases use the algorithm labelled `pairs` in the
archived CSVs and reports. Their old `self_energy` rows used the separate
single-direction algorithm and remain unchanged. Compare new self-energy
timings with old `pairs` rows, taking source revisions into account. No timings
were rerun for this API cleanup.

## Edge-cache lookup

The separate [cache lookup comparison](cache_lookup.md) measures ordered linear,
canonicalized linear and canonicalized hash lookup on matching evaluator calls.
It includes larger key sets and capacity-limited cases.

## Small matrix kernel comparison

```bash
cmake --build cbuild --target dense_few_orb_eigen_gemm --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/dense_cpu_few_orb/dense_few_orb_eigen_gemm > eigen_gemm.csv
```

The experiment compares the production Eigen `small_gemm<N>`, retained
`small_gemm_handwritten<N>`, and NDA/OpenBLAS GEMM on 103 independent complex
matrices for each N=2,4,8,16. Eigen is exercised through the same helper used by
the evaluator, rather than a separately written Eigen expression. It tests
overwrite and accumulation, both compile-time unit scaling (as used in the
evaluator) and general complex scaling. It checks against a long-double
reference, then measures five rotated trials, normally 1024 stack repetitions.
An optional first argument overrides repetitions; a second selects a single
dimension, e.g. `dense_few_orb_eigen_gemm 1024 16`. A compiler barrier prevents
removal of overwritten results. CSV times are nanoseconds per matrix product.

Kernel timings alone do not predict end-to-end gain: the evaluator also performs
DLR transforms, convolutions, edge preparation and output accumulation. Record
compiler architecture flags, Eigen version and BLAS implementation with each
result; fixed-size template code and a BLAS library's runtime CPU dispatch need
not use the same instruction set.

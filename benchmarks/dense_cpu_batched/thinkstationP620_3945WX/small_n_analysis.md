# Why N=4 batching helps modestly, and what helps more

Measured 2026-11-04. The conclusion is not that N=4 cannot benefit from CPU
optimization. The current implementation batches only part of the computation,
and it repeats numerical work that batching does not eliminate. Simple arithmetic
changes already improve complete prepared evaluator calls substantially.

Production source checkpoint: `d13d5ed`, with the user's pre-existing dense
constructor formatting unchanged. Experiment sources and raw results are
committed as `47f9ccd` (`[bench] probe small dense cpu`). CPU dependencies are TRIQS `2930120`, NDA
`4e59077`, and fixed cppdlr `b9b24ab`. Threadripper PRO 3945WX, Clang 19.1.1,
Release `-O3`, generic target architecture (no `-march=native`), system OpenBLAS,
one BLAS/OpenMP thread, CPU affinity 2. Boost remains enabled. Experiments ran
serially, with no concurrent build. This workstation was not otherwise isolated.
No GPU was used and production sources were not changed.

## What batching actually does

A convolution of two `(r,N,N)` arrays performs two coefficient conversions
(four triangular BLAS solves), four DLR-axis matrix multiplications, and three
N-by-N products per DLR node. The batched version shares the solves and DLR
multiplications across `B*N*N` RHS columns. It does not decrease their arithmetic
or the number of small spatial products.

At N=4, even batch one already provides 16 complex RHS columns. Batch eight
widens that to 128, but still makes **3*r*B** separate small BLAS calls inside
each convolution: **2,472 calls at r=103, B=8**. Vertex/endpoint products also
remain separate. The `(tau,batch,row,column)` layout keeps each 4x4 matrix
contiguous; it does not arrange corresponding entries of independent matrices
next to one another for SIMD across the batch.

With conventional complex-operation counts, convolution has leading work
approximately `48*r*r*B*N*N + 24*r*B*N*N*N`, excluding lower-order copies,
pivots and scaling. The small products account for only about **1.9% of these
flops at r=103,N=4**, but much more elapsed time. Flop count alone therefore
misses the cost of issuing thousands of small operations.

The measured production convolution cost per backbone saturates quickly:

| DLR rank | Batch 1, µs | Batch 8, µs/lane | Batch 32, µs/lane |
| ---: | ---: | ---: | ---: |
| 13 | 13.376 | 12.274 | 12.417 |
| 103 | 241.491 | 218.092 | 217.996 |

This is roughly a 9–11% benefit, not a factor of eight. Increasing batch size
does not reduce total work and increases live storage. The measurements support
diminishing returns on this CPU; they do not establish a hardware bandwidth
bound. `perf` access is disabled (`perf_event_paranoid=4`), so the evidence is
source inspection and coarse wall-clock instrumentation, not hardware counters.

## Where time goes

The operation-for-operation convolution clone reproduces production exactly on
the probe inputs. Its separately instrumented batch-eight stage fractions are:

| DLR rank | Coefficient solves | DLR GEMMs | Small matrix products | Scaling |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 36.2% | 18.4% | 41.6% | 3.9% |
| 103 | 38.3% | 40.9% | 19.0% | 1.8% |

Thus small products are disproportionately expensive, but removing them alone
cannot solve the rank-103 case: solves and transforms still take about 79%.

Within complete prepared evaluator calls at batch eight:

| Rank | Kind | Order | Convolution | Vertex multiplication | Endpoint sum |
| ---: | --- | ---: | ---: | ---: | ---: |
| 13 | Self-energy | 2 | 51.1% | 7.7% | 38.5% |
| 13 | Paired self-energy | 2 | 37.0% | 5.5% | 55.1% |
| 13 | Self-energy | 5 | 73.0% | 10.8% | 13.7% |
| 103 | Self-energy | 2 | 70.4% | 4.7% | 23.5% |
| 103 | Paired self-energy | 2 | 56.9% | 3.8% | 38.0% |
| 103 | Self-energy | 5 | 86.0% | 5.7% | 7.2% |
| 103 | Paired self-energy | 5 | 80.1% | 5.3% | 13.4% |
| 103 | Correlator | 5 | 88.3% | 5.9% | — |

These are nonoverlapping categories. Other time includes edge preparation,
reflection/join, output accumulation and allocations. At low order, especially
pairs, the unchanged endpoint sum limits the benefit of batching convolution.

## Two simple arithmetic experiments

**Real DLR arithmetic.** The interpolation matrix, its LU factors, Hilbert
matrix and reconstruction matrices are real. The current helper stores its
tables as complex and calls complex BLAS. View an existing complex `(r,W)`
buffer as real `(r,2W)` and apply real matrices to its real/imaginary columns.
This requires neither a packing copy nor an explicit inverse. The C++ standard
specifies this array access for `std::complex`, and BLAS supports the required
right-side triangular solve. See the [C++ complex representation rule](https://eel.is/c++draft/complex.numbers#complex.numbers.general-4)
and [DTRSM interface](https://netlib.org/lapack/explore-html/d9/de5/group__trsm_ga7120d931d7b1a15e12d50d328799df8a.html).

**Direct 4x4 products.** Replace small BLAS products with a fixed-size complex
multiply loop. The evaluator experiment applies this to convolution, vertices,
endpoints and correlator joins. It uses no reduced precision or fast-math.
The broader principle of amortizing small-kernel dispatch is also reflected in
[LIBXSMM's specialization and dispatch documentation](https://libxsmm.github.io/libxsmm/documentation/libxsmm_tune.html);
no additional library was installed or benchmarked here.

At rank 103, real arithmetic alone improves the batch-eight evaluator by
**1.15–1.29x**; fixed products alone by **1.08–1.15x**. Together they give
**1.39–1.45x over existing batch eight**, or **1.67–1.82x over scalar fast**.
These are complete prepared calls using the existing complex physical model:

| Kind | Order | Backbones | Scalar fast, ms | Current batch 8, ms | Real + fixed4 batch 8, ms | vs current batch 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-energy | 2 | 32 | 24.012 | 19.702 | 13.895 | 1.418x |
| Paired self-energy | 2 | 16 | 14.941 | 12.151 | 8.751 | 1.388x |
| Correlator | 2 | 16 | 11.623 | 9.016 | 6.405 | 1.408x |
| Self-energy | 5 | 63 | 147.107 | 127.531 | 88.255 | 1.445x |
| Paired self-energy | 5 | 63 | 160.193 | 136.416 | 94.804 | 1.439x |
| Correlator | 5 | 63 | 146.303 | 123.897 | 85.904 | 1.442x |

The changes benefit batch one too. Grouping eight backbones still supplies only
about 6–11% over the corresponding improved batch-one implementation. These are
arithmetic improvements rather than evidence that larger batches suddenly scale.

At rank 13, fixed products give **1.18–1.25x** over current batch eight.
Using real GEMMs with fixed products and retaining complex LU gives
**1.22–1.27x**. Replacing the LU solves with real solves is slightly slower at
this small rank; a universal real-solve dispatch needs a rank/shape sweep.
Simply increasing batch capacity remains unpromising. In the standalone
improved convolution, batch 32 is slower per lane than batch eight at rank 103.

## Avoid repeated work before making the batch larger

The edge operand is `G * product(edge.kernels)`. It depends on the current
propagator and the kernel recipe, not on every operator-flavour choice. Current
evaluation reconstructs it and recomputes its coefficients and Hilbert transform
for every use. The untimed census finds substantial exact reuse:

| Self-energy order | Sampled backbones | Edge convolutions | Distinct recipes | Potential call-local reuse |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 32 | 64 | 3 | 95.3% |
| 3 | 63 | 252 | 7 | 97.2% |
| 5 | 63 | 504 | 18 | 96.4% |

These are exact ordered lists, without algebraic simplification. Storing the
coefficients and Hilbert-transformed coefficients costs **51.5 KiB per recipe**
at rank 103,N=4. A hit avoids one coefficient conversion and one Hilbert GEMM,
about one-third of the leading convolution arithmetic. That is an arithmetic
estimate, not a measured evaluator speedup: gathering cached lanes, lookup and
cache capacity still need an implementation experiment.

Keep such a cache scoped to one unchanged `G` and mesh, with a byte limit.
It is evaluator/call data, not immutable Action data. The identity-kernel edge
is an especially simple first case. Cache population must be included in timing.

Consecutive indices improve same-position recipe sharing within a batch. At
order five the stratified sample has 46.8% such reuse, versus 87.3% in the three
consecutive windows. These windows are biased: the first and last use only the
identity recipe. The sample has 18 recipes; larger pole sets or mixed topologies
may have many more. Neither hit rate estimates an impurity solver's workload.

## A larger reuse experiment: fixed-edge convolution matrices

cppdlr can precompute a values-to-values convolution matrix for a fixed left
operand. Application becomes one GEMM. This can help small N, but at rank 103
construction and cache size are substantial. The probe includes required batch
packing and compares against the improved real/fixed4 convolution on the same
changing right operands:

| Rank | Batch | Improved fast, µs | Cached application, µs | Matrix setup, µs | Matrix size | Uses to amortize vs improved fast |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 1 | 11.816 | 2.394 | 65.4 | 42.25 KiB | 7 |
| 13 | 8 | 84.444 | 15.267 | 65.2 | 42.25 KiB | 1 batch |
| 103 | 1 | 164.749 | 141.690 | 8334.8 | 2.59 MiB | 362 |
| 103 | 8 | 1199.418 | 751.394 | 8347.7 | 2.59 MiB | 19 batches |

Setup uses warmed cppdlr tables; cold setup is separately recorded in the CSV.
Batch-eight amortization counts batch applications, each containing eight
backbones with exactly the same edge. It therefore needs about 152 such edge
uses at rank 103. Actual recipe grouping, partial groups, lookup and eviction
are excluded. Right-edge traversal is not prototyped. Cached matrices occupy
`16*r*r*N*N` bytes each; caching all 18 sampled recipes at rank 103 would take
about 46.6 MiB before scratch, risking cache pressure absent from this one-matrix
microbenchmark. Direct per-recipe coefficient caching is the simpler first step.

## Suggested order of further work

1. **Keep the scalar baseline simple and remove its remaining small allocation
   costs.** Mirror the batched implementation's reusable output GEMMs and direct
   correlator trace where useful. The original batch-one gain partly comes from
   these changes and fewer returned results, not from grouping.
2. **Adopt real DLR transforms and a small fixed-size matrix kernel after broader
   correctness/crossover checks.** Keep the existing fallback; dispatch once at
   a suitable outer level. Test N=2/4/8/16 and intermediate ranks before deciding
   which solves/products should use which kernel. No new evaluator hierarchy is
   required for these arithmetic changes.
3. **Prototype a bounded edge-coefficient cache.** Measure cold/warm calls,
   hit/miss counts and retained bytes, using complete bounded topologies and
   realistic pole counts. Then consider grouping equal-edge lanes and selected
   convolution matrices only if reuse pays for their setup.
4. **Target the endpoint sum for low ranks/orders.** A tiny-N linear map on
   `vec(T)` could replace repeated operator products and flavour mixing, or use
   exact sparse/diagonal operator structure. Include precomputation and memory;
   this has not been prototyped. Generic retarded operators need a fallback.
5. **Consider SIMD across backbones or bounded prefix reuse later.** A layout
   such as `(tau,row,column,batch)` preserves contiguous DLR RHS storage while
   exposing lane-wise SIMD for small products, but changes several evaluator
   loops. Exact prefix counting finds about 21% reuse of sampled order-five
   initial states, not enough evidence to justify a large planning/cache layer.
   Do not materialize a combinatorial tree of all contributions.

The next numerical checks should include near-cutoff poles/boundary layers,
multiple beta/rank choices, complex retarded operators, both traversal
directions, and high-order cancellation. Performance follow-ups should include
complete topologies, a solver iteration, several pole/flavour counts, an ISA-tuned
build as a separately labelled experiment, and other CPU/BLAS implementations.
The current trials establish local opportunities, not a portable dispatch table.

## Validation, raw data and limits

- [Convolution stages](convolution_stages.csv): 42 rows, six experimental policies
  plus production, ranks 13/103, batches 1/8/32. Maximum candidate difference
  from production **1.33e-15**. The independent analytic discrepancy is about
  **1.36e-7 at rank 13** (mesh tolerance 1e-6), and about **9.1e-14 at rank 103**
  (tolerance 1e-12). Analytic accuracy and cross-implementation agreement are
  different quantities.
- [Full prepared evaluator](small_n_evaluator.csv): **228 rows**, 12 cases and
  19 variants, maximum difference from scalar **8.16e-14**. Includes first/reused
  calls and partial batch tails. These timings exclude Action/evaluator/backbone
  construction. They are not whole topology or solver timings.
- [Evaluator stages](evaluator_stages.csv): coarse, separately instrumented
  phases. Disabled instrumentation changes the production-like clone's median
  by about 0.23% across cases, at most about 1%; the arithmetic clone is about
  0.8% slower in the median. These small costs do not explain the main gains.
- [Cached operator](convmat.csv): 24 rows. Finite values and direct cppdlr,
  production, improved, and aliasing checks pass at a 2e-10 threshold; errors
  for the smooth probe are around 1e-15. This is not enough to adopt a cached
  operator for arbitrary high-order/boundary-layer data without further tests.
- [Recipe census](recipe_reuse.csv): 36 untimed cases. Reuse is a count of exact
  descriptions, not a timing prediction.

Full evaluator timing uses five rotated trials. Median `max/min-1` spread is
0.61%, with maximum 4.21%. Standalone convolution also uses five rotated trials;
cached operators use seven. No confidence intervals or hardware-counter claims
are made. The production test suite was not rerun because this investigation
only added isolated benchmark targets, prototypes, measurements and documentation.

See the [experiment instructions](../experiments/small_n/README.md) for exact
commands and timing boundaries. Production source remains the baseline for a
separate, reviewable implementation decision.

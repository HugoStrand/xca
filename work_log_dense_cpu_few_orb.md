# Dense CPU evaluation for few orbitals

## 2026-11-04 — original evaluator buffer reuse

The original dense evaluator now writes vertex and endpoint matrix products into
existing storage through GEMM. Vertex evaluation exchanges ownership with the
completed edge buffer instead of copying a temporary product back over its input.
Correlator joins reuse the edge and endpoint-sum buffers; the last trace contracts
matrix entries directly rather than constructing an entire product matrix.
No additional persistent numerical buffer is needed, and the public API is unchanged.

Review checked that no view survives an owner swap, that the completed forward
correlator segment remains untouched during backward evaluation, and that paired
self-energy operands remain distinct. A discovered validation gap was fixed:
prepared calls now reject inconsistent propagator/observable shapes before an
owning scratch buffer can resize. The regression test checks recovery immediately
after invalid calls and permits empty operator lists.

The user's existing constructor formatting/comments and reset spacing are
preserved in the working tree and excluded from the code commit.

Validation: the complete rebuilt CPU suite passed **88/88** entries, including
prepared-call recovery, dense/sparse comparisons, Python integration and actual
three-rank MPI tests. The source change was reviewed separately from the new
evaluator before committing. Existing user formatting remains uncommitted.

## 2026-11-04 — separate few-orbital evaluator

`dense::DiagramEvaluatorFewOrb` retains the same typed-backbone and MPI-local
one-topology responsibilities, with span overloads for sums over prepared
backbones. It is a separate C++ implementation; the original dense evaluator
remains available and Python selection is unchanged.

Its dimension-four matrix products use handwritten real/imaginary arithmetic.
Other dimensions retain BLAS. DLR interpolation solves and transforms use real
matrices on the interleaved real and imaginary columns of packed complex arrays.
This carries forward the arithmetic improvements established by the earlier
small-N investigation; no dimension/rank crossover is guessed from unmeasured cases.

The edge cache stores coefficients of `G * kernel_product` and their Hilbert
transform, keyed by the exact ordered factor list. It shares these operands
across backbone evaluations and traversal directions within a public call. Every
new call invalidates numerical entries, including when G was mutated at the same
address. Allocation slots remain reusable. A default 8 MiB budget bounds cached
numerical payload; zero disables retention, and a full cache falls back to
uncached evaluation for new recipes. This provides an explicit control to measure
the reuse gain separately from the new arithmetic.

Review covered key completeness, left/right multiplication order, reference
lifetime across cache growth, scratch separation, and finite-input beta-zero
semantics. Tests cover cache limits and fallback, repeated calls and changed G,
mixed orders/topologies, both self-energy APIs, reflected fermionic/bosonic
correlators, high-rank order-five samples, N=2/N=8 fallback, strided inputs,
independent outputs, invalid inputs, and MPI partitioning without collectives.

Validation: the new evaluator, kernel and MPI cases pass in the full **88/88**
CPU CTest run (34.43 s). The fixed-size kernel tests include complex alpha/beta,
strided views, and beta-zero overwrite of NaN-filled destinations. Independent
review found no additional blocker. Prepared backbone contents remain trusted,
as in the original evaluator; no sanitizer build was run this round.

## Benchmark organization

`benchmarks/dense_cpu_few_orb` compares the updated original, FewOrb with cache
disabled, and FewOrb with cache enabled. Both this benchmark and the existing
batching driver use `benchmarks/dense_common/n4_fixture.hpp`; the physical model,
seed and samples of the earlier benchmark are unchanged.

Complete order-two and order-three topology calls provide the primary comparison
at matching API boundaries. Bounded prepared samples extend to order five; their
ratios also include the difference between per-backbone returned arrays and one
span result. Every timed call prepares its cache numerical values anew; existing
allocations are warmed. Cache hits/bytes and numerical differences are recorded.

A separate optional benchmark compares handwritten 4×4, fixed-size Eigen, and
NDA/OpenBLAS products. Eigen is not used or required by the production evaluator.
Unit-alpha cases are compiled with constant one to match evaluator products;
general complex-alpha cases remain separate controls. Matrix arithmetic is
checked against a long-double reference before timing.

## 2026-11-04 — measurements and conclusion

Measured N=4 at ranks 13 and 103 on the Threadripper PRO 3945WX, single-threaded
and pinned to CPU 2, using Clang 19.1.1 Release without native architecture or
fast-math flags. Builds/tests were stopped before measurements. Five trials
rotate variant order; raw CSV retains median/minimum/maximum and error.

At rank 103, complete topology calls gain **1.69–2.02×** over the updated original.
The uncached implementation gains 1.42–1.49×; caching adds 1.19–1.36× on top.
Rank-13 complete calls gain 1.42–1.64×. Order-five samples at rank 103 gain
1.98–2.06×, with the prepared-span allocation qualification described above.
No median case regresses when adding the arithmetic changes or enabling reuse.

Complete-call cache hit rates are 90.6–99.7%; order-five samples reach 96.0–96.4%.
At most 20 active recipes use 1,054,720 numerical bytes (1.006 MiB). This workload
never pressures the default 8 MiB cache budget. Retained allocations can exceed
the active bytes of a later smaller call; metadata and fixed scratch are extra.
Maximum relative discrepancy across the 90 evaluator rows is 8.16e-14.

The optional Eigen 5.0.1 experiment passes the long-double reference check
(maximum relative discrepancy 2.59e-16). For unit-alpha products, handwritten
loops take 83.7 ns overwrite / 98.3 ns accumulation; Eigen takes 53.6 / 57.3 ns;
NDA/OpenBLAS takes 128.3 / 116.6 ns. Eigen is thus faster at the isolated kernel,
but the production implementation remains handwritten as requested. Its possible
whole-evaluator benefit is not measured here.

The [hardware report](benchmarks/dense_cpu_few_orb/thinkstationP620_3945WX/README.md)
contains tables, provenance and limitations. The generic benchmark README records
reproduction commands and cache semantics. Follow-up measurements should vary
pole diversity, cache budget, dimensions and realistic solver inputs, then compare
Eigen inside the evaluator if desired. Python selection and solver-level timing
are outside this implementation round.

Code changes are committed separately from work logs/documentation. User edits
remain uncommitted. No remote changes or GPU jobs were performed.

## 2026-11-04 — Eigen kernels through dimension sixteen

Code commit `cc4f66f` makes `small_gemm<N>` use Eigen and retains the original
implementation as `small_gemm_handwritten<N>`. FewOrb dispatches dimensions
2, 4, 8 and 16 to precompiled fixed-size products; other dimensions use BLAS.
The diagram traversal, real DLR transforms and edge cache policy are unchanged.

Eigen is now a normal public dependency, resolved through the repository's
`Build_Deps` mechanism and pinned to 5.0.1 for source builds. The supported
installed package family is Eigen 5. Its headers and CMake package install with
XCA when built from source; the exported XCA target finds and propagates Eigen.
A temporary installation under `/tmp` successfully configured, built and ran an
external C++ consumer. The usual install prefix was restored, and the existing
installed XCA application was not changed.

The packed path uses unaligned fixed-size Eigen maps directly. A standalone
allocation probe uncovered an Eigen 5 issue in the nonpacked path: complex GEMM
can emit a 16-byte aligned scatter store even though a complex-double pointer
only needs 8-byte alignment. That path now computes into a fixed-size stack
matrix and copies back. It reads the previous destination only when beta is
nonzero. The evaluator's packed path remains direct. A deterministic regression
uses pointers at 8 modulo 16, testing both packed and strided cases through N=16.

Validation: **88/88** CPU CTest entries passed in 35.04 seconds. Coverage includes
both kernels, complex alpha/beta, NaN-filled overwrite destinations, padded and
unequal-stride views, N=2/4/8/16 evaluator calls and bounded N=32 BLAS fallback.
The standalone probe with Eigen runtime allocation assertions passed all four
dimensions for packed/strided overwrite, accumulation and general complex
scaling, with no Eigen heap allocation during the products. Code and dependency
integration were reviewed independently before committing.


## 2026-11-04 — Eigen dimension sweep

Benchmark commit `b46b4dd` extends the shared fixture and both benchmark drivers
to N=2,4,8,16 (one through four fermion modes), preserving the original N=4
model. The kernel driver now measures the actual production Eigen helper,
retained handwritten loops and NDA/OpenBLAS products. The evaluator driver
compares the improved original, uncached FewOrb and default-cache FewOrb at
ranks 13 and 103, using complete order-two topologies at every dimension,
complete order three at N=2/4, and bounded prepared samples through order five.
Larger complete order-three cases require `--full-order3` and were not run.

The timings used one thread pinned to CPU 2, Clang 19.1.1 Release and Eigen 5.0.1,
with no native-architecture or fast-math flags. Builds/tests were stopped during
measurements. Every timed call includes numerical cache preparation, while
allocations are warmed. Eigen and handwritten code use the generic target;
OpenBLAS dispatches to CPU-specific kernels at runtime. The prepared comparison
also includes per-backbone result allocation versus one FewOrb span result.

For complete order-two calls, cached FewOrb gives **2.13–2.50×** speedup at N=2
and **1.51–1.93×** at N=4. N=8 gives **0.72–0.84×** at rank 13 and
**0.96–1.27×** at rank 103. N=16 gives **0.58–0.66×** and **0.75–1.00×**,
respectively. Thus extending the Eigen specialization does not improve every
dimension. The kernel benchmark finds Eigen fastest at N=2/4; OpenBLAS is
2.1–2.4× faster at N=8 and 2.9–3.3× faster at N=16 for unit-alpha products.
The requested Eigen default through N=16 remains in place.

At N=16/rank 103, the order-five samples fill the 8 MiB cache with nine entries,
with hit rates 81.5–82.5%. Speedups are 1.08× for self-energy, 0.98× for paired
self-energy and 1.16× for correlators. Complete order two needs only three
entries but still regresses, showing that cache capacity is not the sole issue.
A useful next experiment is to preserve FewOrb arithmetic and reuse while
selecting BLAS only for N=8/16 products. A separate cache-budget comparison and
native-architecture build would isolate additional causes. Neither experiment
was added in this round.

Independent audit checked **324 evaluator rows / 108 cases** and **48 kernel
rows**, including finite values, complete variant sets, timing bounds, diagram
counts, speedup arithmetic and cache accounting. Maximum relative errors were
**2.19e-13** and **7.32e-16**, respectively. The earlier implementation validation
remains **88/88** CPU CTest entries; no source changes followed that test run
other than benchmark code and documentation.

Tables, provenance and limitations are in the updated
[hardware report](benchmarks/dense_cpu_few_orb/thinkstationP620_3945WX/README.md).
The previous handwritten N=4 report is preserved as `handwritten_n4.md`, with
its raw results unchanged. Code, benchmark/results and documentation commits
remain separate. User edits are preserved; no GPU jobs or remote pushes ran.

## 2026-11-04 — shared real convolution and product crossover

Code commit `64a34f73` moves the real DLR arithmetic into the common
`cppdlr::fast_convolve`. Dense, batched dense, block-sparse and FewOrb now use
real DLR tables, real GEMM transforms, and real LU triangular solves on the
interleaved real/imaginary components of complex arrays. No real/imaginary
packing copies or extra owning numerical representations are introduced.
Existing output parameters, reusable workspace and input/output aliasing
contracts remain supported, including changing packed rectangular shapes and
active batch sizes. Integer limits now account for both real lanes.

FewOrb's duplicate DLR tables, transforms, solves and convolution formula were
removed. The shared helper exposes `prepare_operand` to fill coefficients and
their Hilbert transform, and `convolve_prepared` to combine them using a caller's
matrix-product function. The same internal convolution formula serves ordinary,
batched and prepared calls. Edge-cache ownership and policy remain in FewOrb.
The prepared arrays must use the receiving helper's DLR grid; output/product
scratch must be disjoint from the prepared operands.

The measured kernel crossover selects fixed-size Eigen products for N=2/4 and
BLAS for other dimensions, including N=8/16. Explicit Eigen and handwritten
kernels through N=16 remain available for comparison. This is a practical CPU
choice based on the current generic-architecture build; it is not a universal
crossover for every compiler or processor.

Validation: a complete CPU rebuild and **88/88** CTest entries passed (34.68 s).
New coverage reuses a complex rectangular prepared operand on either side of a
convolution, checks the independent cppdlr reference and cache immutability,
and checks inactive tails of packed scalar and batched buffers. Existing tests
cover analytic convolutions, in-place coefficient solves, output aliasing,
changing rectangular dimensions, FewOrb cache budgets and N=2/4/8/16/32 paths.
Independent adversarial review found no blocking issue. The prior claim that
FewOrb uniquely optimized scratch handling was too broad: the original already
reused its workspace; this refactor shares arithmetic and enables prepared
operands without attributing an unmeasured separate scratch speedup.

The next comparison uses the updated original as baseline, FewOrb without edge
reuse, and FewOrb with the same 8 MiB cache. All three share real DLR arithmetic.
At N=8/16 both evaluator implementations use BLAS products; remaining uncached
differences reflect traversal/workspace scheduling and overhead, not Eigen
versus BLAS. Cached versus uncached FewOrb isolates edge reuse. Existing results
are preserved, including the earlier Eigen-through-N=16 report.


## 2026-11-04 — shared-baseline measurements

Results commit `5aa1cc6a` records the unchanged dimension benchmark driver
(`b46b4dd`) against implementation `64a34f73`. Measurements use the same pinned,
single-threaded CPU setup, DLR ranks 13/103, N=2/4/8/16 and default 8 MiB cache
as the earlier sweep. No other builds, tests or benchmarks ran during timing.
The previous CSVs remain unchanged, and the previous report is retained as
`eigen_all_dimensions.md` alongside the earlier handwritten N=4 report.

For complete order-two topology calls at rank 103:

| N | Product selection | Without reuse vs baseline | With reuse vs baseline |
| ---: | --- | ---: | ---: |
| 2 | Eigen | 1.42–1.67× | 1.77–2.08× |
| 4 | Eigen | 1.16–1.23× | 1.50–1.52× |
| 8 | BLAS | about 0.99× | 1.15–1.27× |
| 16 | BLAS | 0.99–1.00× | 1.13–1.26× |

At rank 13, no-reuse gains are 1.63–1.90× at N=2 and 1.27–1.32× at N=4;
with reuse they reach 2.05–2.28× and 1.49–1.57×. N=8/16 uncached cases retain
0.2–2.1% overhead across both ranks with identical BLAS product selection;
this is no longer the earlier large Eigen penalty. Reuse makes every complete
order-two case faster than the updated baseline. The order-five samples gain
1.96–2.38× at N=2, 1.58–1.69× at N=4, 1.22–1.39× at N=8 and 1.21–1.34× at N=16
across both ranks, with the prepared-API allocation qualification retained.

At N=16/rank 103, order-five samples still reach the nine-entry cache limit
(7,593,984 payload bytes); hit rates are 81.5–82.5%. These bounded samples benefit
from reuse despite capacity pressure. Complete order two needs three entries
and does not encounter that pressure.

The baseline also receives the general DLR optimization. As an indicative
cross-run comparison, N=4/rank 103/order-two self-energy changes from 21.362 ms
to 16.898 ms; the main report compares all three new variants within this run
instead of crediting that shared improvement to small products or edge reuse.
Existing isolated kernel measurements justify the N=4 crossover; they were
not rerun because the explicit product helpers did not change.

Independent audit verified **324 rows / 108 cases**, including finite values,
expected variants/work counts, speedup arithmetic, cache accounting and all
reported tables. Maximum relative discrepancy is **3.31e-14**, below 2e-10.
The implementation validation remains **88/88** CPU tests. The
[hardware report](benchmarks/dense_cpu_few_orb/thinkstationP620_3945WX/README.md)
contains full order-two timings, order-three summaries, order-five samples,
cache statistics and limits. No Python API, GPU code, remote state or user
formatting edits were changed.

## 2026-11-04 — one self-energy path

Implementation commit `307ee023` makes the existing paired algorithm the sole
dense topology operation, `compute_self_energy(G, topology)`. It evaluates the
prefix once, contracts both directions of the special arc connected to zero,
and propagates their combined result through the common tail once. Reversing
that arc changes the global self-energy sign, which explains the subtraction
inside `eval_self_energy_pair`. This optimization and the compact work count
are documented beside the implementation and `count_self_energy_pairs`.
Block-sparse evaluation still treats directions separately because their sector
paths and intermediate dimensions can differ.

The baseline's prepared C++ API now names its semantics explicitly:
`compute_self_energy_single_direction(Gt, backbone)` retains the directional
reference, while `compute_self_energy_both_directions(Gt, backbone)` sums both
orientations with all internal choices fixed. Either supplied special-arc
orientation gives the same paired result. Both methods remain hidden from
Python. FewOrb and the experimental Batched evaluator retain only the paired
prepared operation; their directional overloads and runtime `paired` branches
are removed. Prepared spans sum both directions for every supplied backbone,
so a complete topology requires only one representative per pair.

Python's separate `order_by_pairs` method and binding are removed. The existing
solver path now uses paired dense evaluation, with reduction still delayed
until after the topology/order sum. Dense MPI work is partitioned by compact
pairs; in particular, first order is one task. Block-sparse work partitioning
is unchanged. The shared C++ test adapter explicitly selects the baseline's
single-direction method, preserving dense-versus-block-sparse comparisons at
the individual backbone level. Pair equality, opposite orientations, analytic
references, workspace reuse, and independent MPI direction sums remain covered.

Validation: the CPU library and regenerated Python wrapper built successfully;
**88/88 CTest entries passed** in 33.76 seconds, including C++ and Python MPI
tests. The FewOrb, Batched, generated evaluator probe and recipe-reuse benchmark
targets also compiled. Independent reviews found no correctness issues in the
signs, enumeration, scratch/cache handling or caller migration.

Benchmark commit `da2519bc` migrates current drivers to compact self-energy
pairs. Their `self_energy` workload now corresponds to historical `pairs` rows;
the READMEs explain the distinction. Existing raw results and tables remain
unchanged, and no new timing sweep was run. User constructor-formatting edits,
`.gitignore` changes and pre-existing untracked files remain outside these commits.

## 2026-11-04 — batched destination reuse

Implementation commit `6f35c96e` removes the remaining two destination-storage
differences from the batched evaluator. Correlator joins now borrow one matrix
from the completed edge scratch instead of allocating a product matrix for each
batch. Topology wrappers pass their final output directly to the accumulating
self-energy/correlator functions, avoiding a freshly allocated and zeroed
partial result and its extra accumulation for every batch.

The public prepared methods still return independent owning results. Direct
topology accumulation changes floating-point grouping, so numerical comparisons
remain tolerance-based. The existing topology test now covers capacities
1/3/8/19, partial tails, changed propagators, alternating self-energy/correlator
calls and retained outputs. Independent review checked scratch ownership,
homogeneous topology batches, validation on idle MPI ranks and probe hooks.

Validation: all three focused CTest entries passed (batched evaluator, serial
MPI evaluator and three-rank MPI evaluator; 1.08 seconds). The batched benchmark
and generated diagnostic evaluator also built successfully. No new timing sweep
was run; the purpose is a consistent implementation for future comparisons.

## 2026-11-04 — canonical edge-cache lookup

Implementation commit `cf28480c` makes FewOrb cache keys independent of scalar
factor order. It copies each edge's factors into reusable scratch and sorts
`(pole_index, argument_sign)` pairs, retaining multiplicity. Thus `A*B` and
`B*A` share an operand while `A*A*B`, `A*B*B`, and reflected factors remain
distinct. Numerical operands use that same sorted order; input backbones are
unchanged. This is numerically equivalent, not a bitwise-preserving change.

A standard `std::unordered_map` replaces the linear scan. Full vector equality
resolves hash collisions; stored entry indices survive numerical-storage vector
growth. The map owns each key, avoiding a second copy in the operand. No public
API or diagram type changes, cache replacement scheme, or small-cache threshold
are introduced. Numerical arrays and hash buckets persist across public calls;
map contents are cleared, and nodes/keys are rebuilt as needed. Existing cache
budgets still bound numerical payload only.

Benchmark commit `3a33b833` adds an optional controlled comparison of ordered
linear, canonical linear and production hash policies. Generated benchmark-only
controls retain the current numerical evaluation. Twenty cases cover N=4,
ranks 13/103, two/ten poles, complete order-two calls and 512-backbone order-five
samples, 8/64 MiB budgets, and disabled/single-entry caches. Two additional
4096-backbone samples exercise larger key sets. Seven rotated trials include
fresh numerical-cache filling on every call, with warmed numerical storage.

There is no material low-order regression: the new complete order-two medians
range from 0.04% slower to 0.71% faster than the previous lookup. The two-pole
order-five cache shrinks from 31 to 15 entries. The larger ten-pole sample
shrinks from 1,161 to 487 entries (7.37 to 3.09 MiB), with roughly 3% faster
evaluation. At rank 103 with 8 MiB, misses fall from 683 to 336 per 512-backbone
call and evaluation improves by about 3.6–3.8%. Hashing itself adds only about
0.1–0.6% over canonical linear lookup in the ten-pole cases; canonicalization
provides most of the measured benefit. These results favor keeping one simple
hash policy, without claiming a large acceleration from hashing alone.

Validation: the full CPU build succeeded and **88/88 CTest entries passed** in
33.74 seconds. A new regression covers commuting permutations, repeated powers,
sign distinctions and unchanged input backbones for both evaluation kinds.
Existing tests cover cache limits, changed propagators, strided inputs and
repeated calls. All **66 benchmark rows / 22 cases** agree with the baseline
within **9.42e-14** relative maximum error (limit 2e-10). Independent reviews
checked implementation safety, benchmark fairness, tables and cache accounting.

The [hardware report](benchmarks/dense_cpu_few_orb/thinkstationP620_3945WX/cache_lookup.md)
links the raw results and reproduction instructions. The high-order cases are
sampled fully crossing topologies, not full high-order diagram sums. User edits
to the baseline evaluator and `.gitignore`, and existing untracked work, remain
outside these commits.

## 2026-11-04 — Python dense default

Implementation commit `c6378e37` exposes `DenseDiagramEvaluatorFewOrb` through
the existing `_triqs_xca` module and selects it in `Solver` whenever
`conserved_operators=[]`. `DenseDiagramEvaluator` remains the separately
importable baseline. Block-sparse selection is unchanged.

FewOrb now has the baseline's physical-input constructor families, for real
and complex `atom_diag`, with and without retarded interactions. They build
an Action internally and move it into the evaluator. c2py constructs the
evaluator directly, avoiding an Action or evaluator return-value copy at the
Python boundary. An optional `cache_bytes` argument defaults to 8 MiB and
accepts zero to disable retention. The ordinary Solver uses that default.

The wrapped surface contains topology self-energy, single-particle Green's
function and symbolic correlator evaluation. Keyword names `G_ppsc`, `ops_tau`
and `ops_0` match the baseline. C++ Action access, prepared-backbone/span methods,
numerical correlator overloads and cache-statistics access remain hidden from
Python. Topology methods still return rank-local contributions; final reductions
stay in the Python solver. Generated wrappers were regenerated with clair and
reviewed alongside the source declarations.

Validation: the full CPU build and **89/89 CTest entries passed** in 33.49 seconds.
The added Python test covers all four constructor families, keyword calls,
default/disabled caches, order-one/two self-energies, bosonic and fermionic
correlators, changed propagators and independent retained outputs. Existing
serial and three-rank Python tests now assert FewOrb is the dense default.
The generated cache-comparison benchmark also rebuilds successfully. The CPU
installation at `~/apps/xca` was updated, and a fresh installed-package import
constructed a FewOrb Solver and evaluated a nonzero finite self-energy.
Binding regeneration was returned to OFF after validation.

No new timing sweep was needed for this integration. Existing matched
order-two paired-self-energy measurements favor cached FewOrb: approximately
2.08–2.28x at N=2, 1.49–1.50x at N=4, 1.08–1.15x at N=8 and 1.07–1.13x at
N=16 across ranks 13/103. This is not a universal performance guarantee:
uncached N=8/16 cases have measured 0.2–2.1% overhead. At rank 103, the 8 MiB
budget holds nine entries at N=16, two at N=32, and none at N=64 (12.875 MiB
per entry). Larger dimensions, weak reuse and small MPI work shares can lose
the reuse benefit; their crossover needs measurement before adding automatic
fallback policy. First-order diagrams have no convolution edges and therefore
no cache benefit, but no first-order slowdown has been established.

An independent review checked constructor ownership, wrapper signatures,
MPI behavior and test coverage. User edits and existing untracked files remain
outside the implementation and documentation commits.

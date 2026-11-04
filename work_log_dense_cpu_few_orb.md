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

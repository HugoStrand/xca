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

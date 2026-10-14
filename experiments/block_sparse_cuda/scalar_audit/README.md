# Completed-lane cutoff audit

Private diagnostic targeting GH200-class evaluator design. Local RTX A4000 runs
characterize numerical differences; this frontend is not a performance benchmark
or a production selector. All older candidates remain intact.

Compile the streams candidate with XCA_EXPERIMENT_SCALAR_AUDIT. Before filtering,
recompute every completed lane through its general CUDA vertex/edge/special path.
Record raw magnitude/value error and cutoff disagreements inside/outside the old
1e-26 band. Normal mode only replaces lanes selected by that old band, preserving
its failures. --repair replaces every lane and checks exact output/count agreement
with an independent general evaluator. Both use Ordered sparse execution.

Audit copies reference inputs once per cohort and drains existing owned streams.
Its extra reference-input/lane storage is already charged by cutoff replay admission.
The observer is thread-local and reset per frontend call; no production API/bindings
expose it. Check every lane entry for nonfinite values rather than trusting max()
to propagate NaNs. General CUDA is an execution reference, not exact arithmetic.

Build block_sparse_scalar_cutoff_audit with benchmarks/native kernels. run.py runs
four bounded fixtures with/without full repair: rank23 smooth order2, rank36
oscillating nodes, rank47 complex nodes, rank23 smooth order6. It locates one active
path in at most64 metadata indices, calibrates homogeneous scaling, then performs
54 bisections and adjacent-double probes (59rows/case). Each process has120s timeout;
failed output is retained and the campaign stops. Rank follows the declared DLR
window/tolerance. Difficult node patterns are finite API inputs, not claims of
physical propagators or of DLR approximation accuracy.

The fixed band fails on the rank36/47 cases:46 matched-backend count disagreements
remain outside the band. Full reference replacement fixes all counts and reproduces
output exactly in this isolated-lane cohort. It performs general work for every
lane and cannot establish an optimized replay policy. Widening to another empirical
constant is not a conservative error certificate. Next derive/qualify a bound or
eligibility/replay rule before production integration; retain these counterexamples.

`block_sparse_scalar_stage_audit` isolates DLR solve and convolution effects on
three matching node-pattern fixtures. It compares1/512-column solves and coefficient
injection into the same narrow general convolution. Extended64-bit-mantissa host
accumulation measures residuals against the sampled double C matrix; the host
cached-LU inverse supplies a condition estimate. Neither is an interval certificate.

It also compares literal complex products with the general path's BLAS products,
retaining the same rank transforms. Flattened1×1 strided-batched products preserve
independent lanes; broadcast coefficients must match across all512lanes before
same-coefficient conclusions are accepted. The narrow BLAS-product variant matches
general convolution exactly on all three fixtures. The wide variant still differs:
solve width and rank-transform layout remain numerical effects. Product replacement
alone is not a fix or a promoted evaluator. Timings of this diagnostic are excluded
from speedup claims. Outputs/residuals/conditions and memcheck live in hardware folders.

Optional `block_sparse_scalar_stage_audit --sensitivity output.csv` evaluates the
cached-table convolution formula and its analytic Jacobians using224/320binary-bit
arithmetic. Directional and bilinear identity checks validate the formula; precision
agreement must be better than1e-25 without relaxing that criterion. Earlier160-bit
calculation failed this check on cancellation-heavy inputs, so precision was increased.
The node Jacobian multiplies the coefficient Jacobian by the existing double cached-
LU inverse. That inverse remains approximate; convergence is not interval certification.

Sensitivity norms use componentwise complex l1 and row-sum norms. Absolute scales
expand the intermediate terms before cancellation. They distinguish a benign smooth
case from enormous cancellation in the difficult node inputs. All previous42GPU stage
rows must remain unchanged. These are numerical diagnostics; they neither implement
an eligibility rule nor bound the opaque library's floating arithmetic.

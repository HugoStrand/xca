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

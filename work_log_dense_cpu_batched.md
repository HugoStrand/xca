# Dense CPU batching work log

## Batched convolution — 2026-11-03

Source: `[cppdlr] batch convolution transforms`.

Extended the existing immutable fast-convolution helper with a rank-four
`(r,batch,rows,columns)` overload. Coefficient solves and DLR transforms flatten
batch and matrix axes into one wider right-hand side; matrix products remain
independent per DLR node and lane. This performs actual shared transforms,
rather than looping over complete scalar convolutions.

Five capacity-sized scratch arrays expose packed active views for smaller
tail batches. There is no workspace cache per batch size. Tests compare
rectangular complex products against scalar and analytic results, reuse
varying batch sizes, exercise legal input/output aliases, and reject invalid
layouts, shapes, capacity and inverse temperature. The focused CTest passed
(0.19 s). Independent review found no blocker in the formula, aliasing or
integer-range guards. Workspace must remain disjoint from inputs/output;
BLAS may use internal storage.

The upcoming separate `dense::BatchedDiagramEvaluator` is a C++ CPU experiment
with prepared-backbone spans and rank-local topology calls. Preserve the scalar
evaluator. Measure batch one as a control: its output GEMMs and join code also
differ from the scalar implementation, so scalar-to-batch-one gains cannot be
attributed to shared DLR transforms. General correlators use prebuilt numerical
observable matrices; no new Python wrapper or atom_diag conversion API is
needed for the experiment. Existing allocating reflection remains; there is
no claim of allocation-free full evaluator execution.

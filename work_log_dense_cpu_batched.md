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

## Batched evaluator and integration validation — 2026-11-03

Source: `[dense] batch backbone evaluation`.

Added `diagram_evaluator_batched.hpp/.cpp` with bounded batches of compatible
typed backbones, prepared self-energy/pair/correlator sums, and MPI-local
topology APIs. Consecutive changes in order or distinguished-vertex position
end a batch. Partial batches use packed views. Independent operators, kernel
factors and endpoint orientations remain per lane; the DLR transforms are
shared. General correlator topology calls accept numerical operator matrices.

Focused scalar/batch comparisons passed for N=4 with complex hopping,
hybridization and retarded density interactions, both endpoint directions,
both observable statistics, changing cohorts, and capacities 1/3/8/19. Real
three-rank tests also passed for local/reduced sums, empty ranks, validation
on idle ranks and rank-zero-only calls proving no hidden collectives.
Two independent reviews found no sign, aliasing or lifetime blocker.

The broad run exposed a separate existing comparison at 1e-13 relative
tolerance. A temporary renamed dense evaluator built from `59c7bbd` established
that fixed dense and sparse shared almost exact arithmetic, while fast dense
differs by 2.32e-13 (self-energy) and 1.61e-13 (correlator) on that sixth-order
fixture. Across orders 2--6 the maximum was 7.47e-13. DLR grid refinement
changes those results much more. `[test] account for convolution roundoff`
uses 2e-12 only for cross-engine checks; same-engine identities retain 1e-13
and analytic tests are untouched. The focused regression passed afterward.
Diagnostic CSV and provenance are retained with the scalar convolution results.

The N=4 timing driver measures both scalar-fast and batch-one controls, orders
2/3/5, DLR ranks near 13/103, capacities 1--64 and two complete order-two topology
calls. Measurements run without concurrent compilation. Results follow in the
next checkpoint; no speedup is assumed from the implementation alone.

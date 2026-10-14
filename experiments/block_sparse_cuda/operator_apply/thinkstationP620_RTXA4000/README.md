# ThinkStation P620 / RTX A4000 operator controls

Performance target: GH200-class hardware. Local E3 screening is not a production
dispatch table. Code checkpoint `9561926`. All 24 isolated processes / 432 samples
passed numerical screening and exact repeatability. No build or other benchmark
was run concurrently. First-use times, setup/packing, numerical errors, nnz/row
degrees and explicit storage are in [samples.csv](operator-01/samples.csv).
[summary.csv](operator-01/summary.csv) pairs warm medians;
[manifest.json](operator-01/manifest.json) pins exact commands, source/binary/library
hashes, GPU/CPU/toolchain metadata and per-process outcomes.

The custom ordered CSR route matched the CPU sparse reference bit for bit in all
sampled cases. cuSPARSE and dense products passed 1e-13*reference norm+1e-14. Tests
include complex weighted sums, both multiplication directions, empty rows, single
maps, arbitrary RHS widths and the exact 3003-by-3432 adjacent-sector shape. A
separate memory-check smoke (six modes, six terms, width 17) reported zero errors.

Selected warm per-application times (reuse eight, three samples):

| Modes / terms / RHS / direction | Ordered CSR ms | cuSPARSE ms | Dense cuBLAS ms |
| --- | ---: | ---: | ---: |
| 4 / 4 / 16 / left | 0.0056 | 0.3325 | 0.0172 |
| 4 / 4 / 16 / right | 0.0048 | 0.3370 | 0.0172 |
| 10 / 10 / 16 / left | 0.0369 | 0.3374 | 0.7926 |
| 10 / 10 / 16 / right | 0.0358 | 0.3401 | 0.6943 |
| 14 / 14 / 16 / left | 0.5725 | 0.7863 | 100.5374 |
| 14 / 14 / 16 / right | 0.4797 | 0.7953 | 99.8867 |

The retained cuSPARSE path charges per-node descriptors/queries/synchronization,
whereas the custom route launches once across rank. Thus its advantage includes
launch amortization; this is not proof of a cuSPARSE hardware arithmetic limit.
For large narrow-RHS sparse fields, dense expansion costs substantial payload and
work; it is retained as a control rather than assumed useful for these structures.
The custom route stores CSR and plain transpose, not compact maps or implicit sums.

N=14 uses bounded width-1/16 RHS slabs and no atom_diag diagonalization. This does
not qualify full N=14 rank-function residency, a complete diagram or full Solver.
Packing/setup and first-use costs remain outside warm median columns and are
reported separately. Changed-coefficient/reuse break-even, wider sector-sized RHS,
COO capability, end-to-end candidate evaluation and GH200 measurements remain
pending. Before promotion, integrate the candidate in an isolated evaluator and
apply strict numerical/lifetime/budget/MPI gates and inclusive complete-call timing.
See [methodology](../README.md). Production CPU/CUDA evaluators are unchanged.

## Complete evaluator candidate

[Paired full-call screening](../../../../benchmarks/block_sparse_common/thinkstationP620_RTXA4000/ordered-evaluator-01/README.md)
passed all 15 physical workloads with matching discovery counts, CPU screening and
exact replays. Candidate C++ tests and focused leak/memory checks passed. Python/MPI
promotion, wider RHS and target-hardware gates remain pending.

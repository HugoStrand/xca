# Block-sparse benchmark fixtures

Performance target: GH200-class accelerators. Local RTX A4000 results are development
proxies; dispatch and capacity decisions require target measurements.

`fixture.hpp` defines deterministic physical models with unequal onsite energies,
density-density interactions and, for number/mixed partitions, generic complex
hopping. `occupation` uses automatic scalar blocks and diagonal hybridization;
`number` conserves total particle number with binomial sector dimensions; `mixed`
conserves particle numbers in two hopping-connected orbital groups. Compatible
hybridization coefficients are complex within each current symmetry set. Optional
retarded channels are total density and an orbital-transfer monomial, with diagonal
channel coefficients. The current grouping restrictions are preserved.

Analytic sector inventories need no atomic diagonalization. They report payload
capacity at a requested rank, not physical rank/accuracy qualification. Actual model
inventories use the DLR rank for beta=2, Lambda=40 and epsilon=1e-10. Operator snapshots
use direct Fock CSR export; rank functions remain dense within sectors.

Path counts are structural candidates with an explicit all-live propagator mask.
They include paths whose arithmetic is zero or rejected by the retained CPU's
per-path magnitude cutoff. No numerical evaluation or GPU arithmetic is timed.
Record rejected proposals as well as accepted paths; active-only selection cannot
establish complete Solver throughput.

Shape rows describe matrix geometry. `calls` counts rank batches for vertex products,
three rank batches per convolution, individual rank-r solves with their separate RHS
widths, and special-vertex right/left products per set member. `special_mix` is an
entrywise weighted reduction: K is the set width and calls is the number of mu
members, not a GEMM dimension. These are workload descriptors, not measured FLOP rates.

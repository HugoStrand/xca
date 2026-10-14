# ThinkStation P620 / RTX A4000

`prefix-01` is a CPU structural inventory, not a GPU timing campaign. The GPU
name identifies the development workstation; GH200-class hardware remains the
optimization target.

The recorder completed 24 bounded cases and 72 stage rows. It scanned 3, 15,
38 and 1088 pairings for orders 2, 3, 4 and 6, selecting respectively 1, 4,
6 and 6 connected topologies. No normal case reached the 4096-path cap.
Executable controls check deliberate truncation and malformed input rejection.
The CUDA configuration reproduces the order-4 number inventory byte-for-byte.

For active number-symmetry paths, order 3 has 8 shared edge keys across
4 topologies (20 distinct-topology reuses); order 4 has 8 across 6 (24 reuses);
order 6 has 4 across 6 (20 reuses). Unique edge payloads are respectively
89,792, 54,464 and 38,272 bytes for this rank-23 four-mode fixture. Root copies
are separate. Counts describe candidate identity, not measured saved FLOPs.

The order-6 mixed case has no shared operator/edge keys. All periodic-mask cases
have no cross-topology overlap; some have zero surviving paths. Order 2 has only
one connected topology and serves as a no-cross-topology control. This selection
is deliberately small and biased toward early pairings and flat indices. It does
not predict overlap for all indices, density operators, or ten/fourteen modes.

Numerical qualification, cache lifetime/input invalidation and full-call timings
remain required before any DAG or CUDA graph promotion. Existing evaluator
arithmetic, topology reduction order and backend defaults are unchanged.

`replay-01` qualifies the same 24 cases numerically: 384 repeated-key comparisons
are exactly equal. Independent dense-operator/original-CPPDLR prefixes differ
by at most 8.423845684525094e-18 in absolute value from CSR/hoisted prefixes.
Maximum retained candidate tensor payload is 204,608 bytes across all stages
of a case. Fresh changed-G and changed-coefficient controls change non-root
checksums while retaining exact repeated-key equality. The resource cap and
malformed-input controls also pass. This is CPU numerical qualification, not
a measured GPU sharing gain or a complete-call equivalence check.

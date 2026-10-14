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

`cache-01` measures complete device self-energy cohorts using both Library/Reference
and Ordered/Mapped policies. Eighteen processes / 162 samples cover orders 3/4/6,
three partitions and cold, unchanged replay and per-sample changed-G phases. All
per-topology outputs match independent device calls exactly. Median timings include
snapshot comparisons, cache population/misses/copies and public call transfers;
construction and externally requested cold cleanup are excluded. Changed-G cleanup
is inside timing. Neither MPI nor Python topology summation is timed.

Ordered/Mapped order-3 number symmetry: independent/cached medians are 62.884/49.439
ms cold, 63.214/39.347 ms replay and 63.308/49.422 ms changed G, respectively 21.4%,
37.8% and 21.9% reductions. Order-4 number changed G improves 13.2%; order-6 number
7.1%. Order-3 mixed changed G improves 18.0%, but order-6 mixed loses 0.6% with no
cross-topology prefix hits. The selected occupation order-4/6 calls have no surviving
prefix work and show small overhead; they do not characterize the full occupation
workload. Candidate charges range from zero to 96,800 bytes in these selections.

These are paired three-sample local medians, not GH200 predictions or population
statistics. The first uncommitted timing run was replaced: it changed G only once
per phase and tested only Library/Reference. Final changed-G samples all invalidate
and reconstruct the cache, and both baseline policies are represented.
Eight inherited observable/API/numerical controls pass; Library and Ordered/Mapped
memcheck runs report zero errors and leaks. Cache-cap saturation, wider index ranges,
full topology cohorts and GH200 qualification remain before production adoption.

`selection-01` broadens Ordered/Mapped cohorts: all 15/105 pairings are scanned at
orders 3/4, yielding 4/27 connected topologies, with all four or six evenly spaced
connected selections. Order 6 scans 4096 of 10,395 pairings, yielding 963 connected
members; six are spread across that bounded prefix. This is not population sampling.
First indices 0/7/31 each cover three indices, across the three four-mode partitions;
one additional six-mode number case uses first 7. All 28 processes / 252 samples
match complete outputs and accepted/contributing counts exactly. Maximum cache
charge is 2,278,656 bytes, below the default 8 MiB cap.

Changed-G order-3 active four-mode medians improve 18.0–27.1%; the six-mode number
case improves 21.5% (202.672 versus 159.149 ms). Order-4 active cases improve
11.1–20.3%, while order-6 number cases improve about 8.9–9.0%. Order-6 mixed cases
range from a 0.7% regression to a 1.0% reduction, with zero or two cache hits. The
occupation order-4 first-zero and all selected order-6 cases have zero accepted
paths and are excluded from active-work gain claims. Some order-4/6 number paths
are accepted but fail the output cutoff; both evaluator counts still agree.

Discovery and model construction are outside call timing; input invalidation,
cache key construction, retained copies and transfers are inside it. Sequential
paired medians use three samples and four/six small modes, not GH200 or full large-
sector systems. Nonzero-offset mixed order-4 memcheck and argument-limit controls
are checked separately. Larger random/index cohorts and a selection policy remain
before production adoption. Earlier forced-limit controls cover cache saturation.

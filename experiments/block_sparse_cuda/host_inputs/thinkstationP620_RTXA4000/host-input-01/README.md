# Host-backed inputs: RTX A4000 screen

Source a0ae481. Six cases, three routes, three warm samples each:54 samples.
Route order reverses on alternating cases; one OpenMP/BLAS thread, rank23, one
selected OCA topology/index. Routes: unchanged library/reference evaluator, resident
lifetime and host-backed input. All use resident complete output for timing. Host-
sector output is separately qualified in C++ controls. Transfers/allocation/drains
are included. Hashes/commands/process statuses in manifest.json; every process passes
finite/CPU-screening/exact self-replay checks. Signal/error/structural fields agree
offline across all three routes at ordinary amplitudes.

| Case | Reference ms | Lifetime ms | Host input ms |
| --- | ---: | ---: | ---: |
| N4 number OCA | 32.807607 | 32.563348 | 32.493457 |
| N6 number OCA | 101.466249 | 100.566331 | 101.545528 |
| N8 number OCA | 265.981058 | 265.695833 | 268.862623 |
| N10 number OCA | 1104.346302 | 1110.429653 | 1114.810099 |
| N6 mixed SPGF | 175.578470 | 176.407406 | 176.882758 |
| N4 number dynamic correlator | 18.340681 | 18.352724 | 18.383481 |

N10 host-input declared bound265175200 vs lifetime333165408 vs baseline450012768
bytes (41.1% below baseline); median latency0.95% above baseline. Sampled global
memory deltas226492416/299892736/343932928 bytes, respectively. These are not
continuous/process-local peaks; small cases can show zero because of caching and
granularity. No speedup/default/crossover/whole Solver claim from three local samples.
Host upload reuse can cost much more for full sums/high orders; one index cannot
establish its dispatch policy.

Eleven controls pass native/library-only9.79/9.76 s, both native flags OFF in library
tree. CPU configure omits experiment. Three focused memcheck controls pass27.998 s,
zero errors/leaks. Inherited dynamic/high-order/correlator tests plus exact baseline
controls exercise changed G/masks, complete host output, exact budgets, one-byte-below
rejection, baseline rejection at new budget and invalid-call recovery. All18 adjacent-
cutoff probes at ranks23/28 match production matrices/counts exactly. No new broad
suite, Python/MPI or target qualification. RealBatch arena is explicitly unsupported
by this private host-input constructor; reference NCA remains covered.

Metadata-only capacity formulas: complex tensor T=16*r*D^2, complete function
F=16*r*sum(d_q^2), number sectors d_q=binomial(N,q). Fixed payload comprises five
complex rank tables, integer pivots/status, two r*2 hybridization arrays,32 MiB sparse
reserve and one rank vector. Host-input resident-output OCA F+7T+fixed; complete host-
output OCA8T+fixed; correlator base2F+7T+fixed (endpoint/result/trace extras additional).
No host input mirror is allocated; caller-owned input and full host output remain
live and must be counted on host. Opaque device allocations/free-memory headroom,
host model/atom construction and physical rank accuracy remain separate requirements.

N14/r64 host-input/host-output OCA estimate89.895 GiB is below observed GH20095.58 GiB,
but is **not proof of fit**. Host input/output alone76.516 GiB, before models/temporaries;
the standard CPU-control fixture could exceed node memory and must not be run blindly.
Correlator base155.179 GiB still exceeds device capacity: fragments remain resident.
Rank26 estimates are diagnostics, not physical accuracy qualification. Next prepare a
bounded full-shape phase/map probe with actual free-memory checks and host/device peaks,
then separate fragment streaming. Do not launch a full physical N14 Solver from this
estimate. Preserve resident baselines and row semantic counterexample. GH200-class
target, local A4000 only screen. No Arrhenius allocation used.

# Full-shape lifetime screen: RTX A4000

Timing source 0580dfb; tightened bound/control source b549256. Binary hashes and
commands in manifest.json/qualification.json distinguish these phases. Six cases,
reference/candidate route order alternates by case, three warm samples each:36 rows.
Same library sparse/reference special policies, one selected OCA topology/index,
rank23, one OpenMP/BLAS thread. All process statuses zero, finite/CPU-screening/exact
self-replay checks pass; signal/error and structural counts agree exactly offline.

| Case | Reference median ms | Lifetime median ms | Reference sampled bytes | Lifetime sampled bytes |
| --- | ---: | ---: | ---: | ---: |
| N4 number OCA | 32.755219 | 32.914237 | 0 | 0 |
| N6 number OCA | 102.220154 | 100.828493 | 2097152 | 0 |
| N8 number OCA | 265.659024 | 265.846586 | 29360128 | 23068672 |
| N10 number OCA | 1108.917737 | 1107.884578 | 343932928 | 299892736 |
| N6 mixed SPGF | 175.742779 | 177.465029 | 0 | 0 |
| N4 number dynamic correlator | 18.442983 | 18.362282 | 0 | 0 |

Latency is roughly flat in this small sample. N10 sampled global free-memory delta
falls 12.8%; it is not a continuous/process-local peak, and zeros reflect allocator
caching/granularity. No universal speedup/default/crossover claim. Timing retained
old conservative bounds; tightened admission did not change convolution arithmetic.

Initial native/library tests: ten controls, 9.44/9.27 s. Final tighter-bound tests:
eleven controls, 9.80/9.85 s. Focused memcheck three controls,29.307 s,zero errors/leaks.
CPU configure omits experiment; both native flags OFF in library tree. Final controls
prove complete SE/SPGF at exact private budget, rejection one byte below, rejection
by baseline at that budget, NCA bounds unchanged, changed G/masks/output placement,
invalid input/endpoint recovery and exact baseline matrices/counts. The first build
had one leftover row-cap constructor argument from fixture adaptation; removed and
rebuilt. No runtime failure was hidden. Inherited tests cover dynamic/high-order and
correlator APIs. This is not a new broad-suite/Python/MPI qualification.

Both cutoff audits (before/after bound tightening) compare18 neighboring scales at
ranks23/28. All matrices/errors and contribution decisions match production exactly,
including the scale where row chunking had failed. No reduction shape or cutoff changes.

Generic phase proof lives in the experiment README. capacity-estimates.csv is metadata
only, special backend Reference, two poles,32 MiB sparse reserve; includes full inputs/
outputs, seven temporary rank tensors and fixed tables. Opaque workspace/headroom and
host atom construction remain additional. The five-rank-tensor bound reduction is
not a claim that all five were actually allocated at once. Tight budget controls use
N4; N10 timing uses the old bound. N14/r64 host-output OCA estimate128.154 GiB still
exceeds the observed GH20095.58 GiB; N14/r26 estimate52.081 GiB is no physical rank/
capacity qualification. Further input/fragment residency work is required.

Candidate remains private pending broader qualification and target evidence. Preserve
baseline, row counterexample and all experiments. GH200-class target,A4000 local
screen. No Arrhenius allocation used; tracked cumulative target usage unchanged.

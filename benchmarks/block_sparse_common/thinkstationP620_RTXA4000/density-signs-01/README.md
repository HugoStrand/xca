# Special parity hoisting: old/new matched calls

All18 candidate tests passed47.00s. Four workloads/four rebuilt routes/three samples
produced48 rows, passing finite output, CPU numerical screens and exact per-route
replay. GPU norm/error/count summaries match old/new; max error1.570e-16. This does
not independently assert elementwise bitwise old/new GPU equality. CPU-only new
control passes three N6 samples exactly against originalCPU.

Median complete synchronous API times (ms):

| Case | Old CPU | New CPU | Old GPU | New GPU | GPU speedup | New CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N4 NCA, two indices | 1.732 | 1.145 | 2.812 | 2.231 | 1.26 | 0.51 |
| N4 OCA, eight indices | 6.232 | 4.030 | 6.402 | 4.137 | 1.55 | 0.97 |
| N6 OCA, eight indices | 23.892 | 13.727 | 21.989 | 11.619 | 1.89 | 1.18 |
| N8 OCA, one index | 10.979 | 6.006 | 12.604 | 7.496 | 1.68 | 0.80 |

Hoisting repeated field assignment/parity benefits both CPU and GPU. New GPU wins
over matched CPU only in the sampled N6 case, is approximately even at N4 OCA, and
loses elsewhere. These three-sample local measurements do not establish robust
dispatch or whole Solver throughput. Rank23, explicit three-field retarded set,
one crossing topology and bounded indices; A4000 is a development proxy, GH200 the
target. Setup/warm-up/screening excluded, preparation/transfers/cutoff/fan-in included.

All timing was sequential after tests/builds; CPU-only compatibility build followed
the campaign. No instrumentation or competing GPU work during timing. Device
payload/lifetimes are unchanged; the new bounded host sign vector requires no
additional device admission. Existing cutoff replay limits remain. Raw commands,
binary/runner/source hashes, samples and detailed tests are retained. No target
allocation was used. Next remove repeated weight transfers/synchronizations in a
separate candidate and qualify finalists on GH200 before promotion.

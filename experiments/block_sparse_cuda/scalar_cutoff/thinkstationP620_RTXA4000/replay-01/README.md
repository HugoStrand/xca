# Local cutoff replay evidence

The unreplayed active order-six probe failed at adjacent amplitudes around
0.10775810972137165: reference retained a ~1e-16 contribution, candidate discarded
it. Replayed probe passed. Original constants and new replay CTests passed in
8.81/9.97 seconds (18.79 total). Replay includes 16 active GoogleTests; constants
retains 15. Focused replay cutoff/admission memcheck passed two tests in 12.971 s,
with zero errors and leaks. Individual logs are retained.

A separate unprofiled N=6 scalar order-two range, first=1/count=8, five samples per
route, passed CPU numerical screening and exact replay. Both accept 640 structural
paths and show the same CPU comparison error, 6.381e-17. Median complete-call times
were 7.484 ms for constants and 7.327 ms for replay. This single sequential screen
shows no evident ordinary-call regression; it does not establish a speedup or replay
cost for a cohort full of borderline lanes. Build/tests/memcheck were finished before
timing. GH200 remains the target; no additional target allocation was used.

The fixed uncertainty band is not a general numerical error bound. More poles,
topologies, cancellation and conditioning, full replay-cost/admission measurements
and GH200 qualification remain open before promotion. Production is unchanged.

# Complete-call comparison

Staging candidate versus resident GPU and parity-hoisted CPU. Nineteen controls
passed in46.17s; three focused memchecks passed38.926s, no errors/leaks.
Initial NDA descriptor extent-constructor host crash diagnosed and fixed before
qualification. Buffer reuse retains operation completion and charges12,288 extra
bytes at512 lanes. No profiler evidence was collected for API-count changes.

Four bounded density cases, three routes and three samples each (36rows). Finite
outputs, per-frontend CPU screening and exact repeatability pass. GPU summary
fields agree across variants; this is not a cross-route elementwise bitwise check.
One crossing topology, rank23; setup/warm-up excluded from call timing.

| Case | CPU ms | GPU baseline ms | GPU candidate ms | GPU speedup |
| --- | ---: | ---: | ---: | ---: |
| density-n4-nca | 1.192 | 1.794 | 1.690 | 1.06 |
| density-n4-oca | 4.191 | 3.314 | 3.132 | 1.06 |
| density-n6-oca | 13.787 | 8.741 | 8.013 | 1.09 |
| density-n8-oca | 5.936 | 5.852 | 5.374 | 1.09 |

Local gains6–9% over resident; smallNCA still favors CPU. Target qualification
of staging remains open. RTX A4000 is a development proxy; target is GH200-class.
Experimental cutoff replay band remains unproven outside its tested envelope.
No production/Python/MPI promotion or wholeSolver speedup claim follows.

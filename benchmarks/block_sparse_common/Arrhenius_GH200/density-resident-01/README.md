# Complete-call comparison

Resident-weight candidate versus parity-hoisted GPU and matched CPU. Job3828369
COMPLETED0:0, oneGH200120GB on n123,245s=.06806GPU-hours against15min cap.
Nineteen candidate plus20 production controls passed across fiveCTest entries
in53.07s. Source77f59ac, GTest988ea2c; submitted job/runner hashes retained.
Installed ArmTRIQS reused; no install or shared-checkout changes.

Four bounded density cases, three routes and three samples each (36rows). Finite
outputs, per-frontend CPU screening and exact repeatability pass. GPU summary
fields agree across variants; this is not a cross-route elementwise bitwise check.
One crossing topology, rank23; setup/warm-up excluded from call timing.

| Case | CPU ms | GPU baseline ms | GPU candidate ms | GPU speedup |
| --- | ---: | ---: | ---: | ---: |
| density-n4-nca | 0.430 | 3.280 | 2.923 | 1.12 |
| density-n4-oca | 2.442 | 5.438 | 4.265 | 1.28 |
| density-n6-oca | 8.370 | 10.492 | 8.848 | 1.19 |
| density-n8-oca | 4.760 | 7.165 | 6.153 | 1.16 |

Resident gains12–28% over parityGPU; matchedCPU wins all four cases. Neither a
universal GPU crossover nor performance at full10/14-mode systems is established.
This run qualifies resident only; staging was not included.
Experimental cutoff replay band remains unproven outside its tested envelope.
No production/Python/MPI promotion or wholeSolver speedup claim follows.

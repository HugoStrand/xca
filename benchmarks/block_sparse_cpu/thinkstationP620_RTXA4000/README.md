# ThinkStation P620 workload inventory

CPU: AMD Ryzen Threadripper PRO 3945WX (12 cores); GPU: NVIDIA RTX A4000,
16 GB, driver 580.178.04. This campaign uses the CPU only. GH200-class devices
remain the optimization target.

Run on 2026-10-14 against benchmark commit `de527f1`, with one OpenMP/OpenBLAS
thread. `manifest.json` records source/header/binary hashes, commands, return codes
and host metadata. All 150 cases completed successfully in 56.65 seconds summed
subprocess wall time: 144 actual atom_diag inventories and six analytic inventories.
Each actual case uses order 3, a 32-index window and an all-live propagator mask.
The windows begin at zero, one orbital/direction group and the domain tail, for
both crossing and chain topologies. Raw per-case summaries and shape histograms
are retained; `summary.csv` collects all summaries. Empty stderr files are omitted.

For the first crossing window without retarded channels:

| Modes | Partition | Blocks | Largest block | Accepted / proposed | Null trace (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 10 | occupation | 1024 | 1 | 29696 / 327680 | 4.660 |
| 10 | number | 11 | 252 | 256 / 352 | 0.015 |
| 10 | mixed | 36 | 100 | 1246 / 2304 | 0.065 |

These are three-sample means for descriptor generation, with no numerical arithmetic.
The scalar regime exposes many rejected proposals and accepted tiny operations;
large-sector work has far fewer paths but larger rectangular operations. This does
not establish a kernel winner or a complete Solver throughput comparison. Actual
N=10 number/mixed setup costs approximately 2.4/1.9 seconds per invocation here.

The analytic N=14 total-number case at rank 26 needs 16,688,505,600 bytes for one
all-sector function and has a 4,899,907,584-byte largest rank buffer. Legacy scratch
and initial correlator-side capacities are 88,198,336,512 and 146,997,227,520 bytes;
these are incomplete peak diagnostics, not measured allocations. No N=14 atomic
diagonalization or GPU arithmetic was run.

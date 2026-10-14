# Production real NCA dispatcher qualification

Code cd54781, benchmark frontend 40f1a0e. Focused native integration and primitive CTests pass 54.08 s, CPU entries .03 s (GPU disabled), native-disabled entries .76 s (policy rejection active). Retained streams and real-NCA experiment controls pass 104.17 s. Two integration controls under memcheck pass 5.980 s with zero errors/leaks. No full-suite or Python/MPI integration claim for the new policy.

Controls compare real dynamic multi-set NCA with all four reference operator/special modes, >512 paths and tail, changed complex inputs/recovery, cutoff adjacency, subnormal nodes, unsafe ranges, complex models, higher-order and host-sector fallback, mixed/larger block fallback, SPGF, empty ranges and budget rejection.

Correctness passes, but the first complete-call benchmark finds N=8 production slower than the prototype (8.60 vs 2.15 ms). Separate per-sector allocation versus the prototype arena is the next hypothesis to test. Performance evidence lives in benchmarks/block_sparse_cuda/thinkstationP620_RTXA4000/real-nca-production-01. GH200 remains unqualified for this route; no cluster hours consumed.

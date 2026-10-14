# Matched batched CPU/GPU density-set calls

Four workloads, four routes, three samples: 48 rows passed finite output, CPU
screening and exact repeated-call checks. Matched CPU agrees exactly with unchanged
CPU on these cases; maximum GPU absolute error is 1.570e-16. The same validated
scalar/three-density-field fixture and rank 23 are used, one crossing topology.
Processes ran sequentially without build/test/profiler competition.

Median synchronous API times (ms):

| Workload | Original CPU | Matched batched CPU | General CUDA | Batched CUDA | Matched CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=4 NCA, two indices | 1.151 | 1.754 | 57.908 | 2.821 | 0.62 |
| N=4 OCA, eight indices | 37.382 | 6.333 | 380.428 | 6.374 | 0.99 |
| N=6 OCA, eight indices | 216.848 | 23.985 | 1950.302 | 21.964 | 1.09 |
| N=8 OCA, one index | 129.319 | 11.253 | 845.724 | 12.576 | 0.89 |

Most improvement over unchanged evaluators comes from the batching/transform
algorithm. Against matched CPU, GPU is approximately even at N=4 OCA, modestly
faster at N=6, and slower at N=8. Original N=4 OCA CPU differs noticeably from the
earlier campaign, reinforcing that three samples/one campaign do not establish
robust thresholds. No production dispatch or whole Solver/topology-sum claim follows.
The CPU control shares the recipe/schedule but not the GPU borderline-cutoff replay
policy. These ordinary workloads do not qualify numerical-policy equivalence at
arbitrary boundaries. A4000 is a development proxy; GH200 remains the target.

The matched host control also built without CUDA and passed three CPU-only N=6
samples exactly against original CPU. Default singleton host recipe still screens
exactly. Focused GPU multi-set and adjacent-cutoff controls passed in 2.131 seconds
after backend-independent zero/add forwarding was introduced. No new device memory
policy or allocation lifetime was introduced by that forwarding.

Raw rows, commands, runner/binary/source hashes and compatibility logs are retained.
Matched CPU/batched GPU were rebuilt; original CPU/general CUDA binaries were
retained controls, explicitly identified in qualification.json. Setup/warm-up and
screening calls are outside latency; discovery, packing, ordered sums and output
construction are inside. No additional Arrhenius allocation was used.

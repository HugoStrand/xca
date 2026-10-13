# Larger power-of-two controls

Evidence after reorganization: [CPU/shared baseline](../../dense_cpu/thinkstationP620_RTXA4000/baseline/), [native and larger-dimension runs](../../dense_cuda_small/thinkstationP620_RTXA4000/results/), [Solver workload](solver-workload-current/). Paths in historical commands below refer to the recorded revision.

RTX A4000, rank16, two poles, seed31415, crossing topology. GH200-class cards are the target and still need separate measurements. Eighteen fresh single-process screens compare the native sequential library route and retained CUDA evaluator at dimensions16/32/64. Orders1/3/6/8, ordinary/paired SE and SPGF, counts1/8 at order1 and3 at higher orders, samples3. All screens pass the unchanged individual/aggregate CUDA gate and low-order CPU checks; high-order CPU comparisons remain diagnostic. Fixed-size kernels at these dimensions are not admitted by these library-route measurements.

|N|Order|Method|Indices|Native library(ms)|Retained CUDA(ms)|CPU(ms)|
|---|---|---|---|---|---|---|
|16|1|se|8|0.536|1.121|0.602|
|16|1|pairs|8|1.009|2.175|1.031|
|16|1|spgf|8|0.633|0.651|0.463|
|16|8|se|3|13.745|20.468|31.830|
|16|8|pairs|3|14.480|22.108|32.602|
|16|8|spgf|3|15.434|20.646|32.232|
|32|1|se|8|1.052|1.922|3.200|
|32|1|pairs|8|1.966|3.674|6.215|
|32|1|spgf|8|1.109|1.132|3.405|
|32|8|se|3|21.252|37.476|129.917|
|32|8|pairs|3|22.624|38.720|134.485|
|32|8|spgf|3|23.327|37.407|135.973|
|64|1|se|8|6.431|7.971|22.691|
|64|1|pairs|8|12.312|15.042|44.038|
|64|1|spgf|8|3.985|5.121|30.156|
|64|8|se|3|83.859|105.968|570.970|
|64|8|pairs|3|92.658|112.116|600.680|
|64|8|spgf|3|86.217|112.538|625.031|

This is a screen, not three-process uncertainty or a throughput crossover for large batches. Native ownership/stream/resource policies can improve time even when every matrix product uses libraries. There is no need to replace retained CUDA: both implementations remain available, and their applicability can differ by workload/hardware.

The earlier N16 low-order CPU-disagreement fixture in power_two/results remains a distinct recorded rejection. Passing seed31415 here does not erase it or establish universal numerical equivalence. Larger fixed-size product admission requires separate primitive/full-API qualification and useful gains. No tolerance or precision changed.

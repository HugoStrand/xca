# Real Solver workload control

Serial RTX A4000 control, three fresh Solver objects per backend/case. Ordinary or fitted two-channel density/spin interaction; two self-consistency iterations through order2, SPGF through order2, followed by order2 density correlator. Full fit/setup/solve/observable paths are included, not independent C++ evaluator calls. All resulting G, Sigma, G_tau and density correlators match CPU with the same integration-test tolerance(rtol1e-9, atol1e-12), and are finite/nonvacuous. The strict C++ optimization gate is unchanged.

|Backend|Interaction|Median setup(ms)|Median solve(ms)|Median observable(ms)|Median total(ms)|
|---|---|---|---|---|---|
|cpu|ordinary|2.949|33.257|10.482|46.880|
|cuda|ordinary|3.211|212.148|68.598|284.114|
|cuda_small|ordinary|4.928|44.074|34.624|83.610|
|cpu|dynamic|3.002|116.387|20.595|140.254|
|cuda|dynamic|3.120|880.265|131.965|1015.960|
|cuda_small|dynamic|5.603|109.012|68.714|183.185|

CPU is faster for these short complete workloads; native Small improves retained CUDA substantially, but no overall GPU speedup is claimed here. Larger indexed batches and high orders are represented by the separate C++ benchmarks. Solver defaults therefore stay CPU. Each median includes three samples in one process, not three-process uncertainty; setup/context warm state and fitting costs matter. MPI correctness is tested separately.

The target is GH200-class accelerators. These local timings do not justify target crossover thresholds or an automatic backend policy. Re-run the standalone script in the target environment; retain its manifest/raw samples and any failed comparisons.

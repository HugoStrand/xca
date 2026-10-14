# Packed production real NCA qualification

Code ef59a06. Same dispatcher controls as real-nca-dispatch-01, now packing scalar sectors into two contiguous arenas. Device payload admission still covers full input/output functions and the transient batch workspace. Exception cleanup drains the owned stream before arena destruction.

Nine active native controls pass in two CTests / 53.62 s. CPU-only metadata passes, GPU controls disabled. Native-disabled constructor/policy rejection controls pass, native-only controls skip. Two integration and four primitive controls pass separate serial memcheck runs with zero errors/leaks. A new primitive control rejects partially overlapping input/output views and repeated output storage before submission. No broad full-suite, Python/MPI or GH200 claim.

Benchmarks are in benchmarks/block_sparse_cuda/thinkstationP620_RTXA4000/real-nca-arena-01. The initial per-sector campaign remains as a performance control. Retained experiment regression tests passed before packing; their numerical implementations were not changed. No new Arrhenius job or allocation consumed.

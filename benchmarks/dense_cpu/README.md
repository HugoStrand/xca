# Dense CPU benchmark

CPU-only timings use the same physical fixtures, selected indices, CSV schema,
warmups and first-call accounting as the CUDA-small benchmark. The dedicated
executable always selects CPU execution, even in a CUDA-enabled build.

```bash
cmake --build cbuild --target dense_cpu_benchmark --parallel 6
python3 benchmarks/dense_cpu/run.py --output benchmarks/dense_cpu/<hardware>/results/run-01 -- \
  --orders 1,3 --indices 1,8 --samples 3
```

Configure with `Build_Benchmarks=ON` and load the environment matching the build.
The output directory must be new. See the [shared methodology](../dense_cuda_small/README.md)
and [comparison tools](../dense_comparison/README.md).

[Workstation evidence](thinkstationP620_RTXA4000/README.md) includes the original
CPU baseline and prefix controls. Historical shared baseline runs contain both CPU
and CUDA rows; keep the raw records together and select the CPU backend when comparing.

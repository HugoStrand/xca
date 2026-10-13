# Dense CPU benchmark

The CPU entry point uses the [shared physical fixture and timing/CSV contract](../dense_common/README.md).
It always selects CPU execution, including in CUDA-enabled builds, and includes
no GPU adapter. Configure with `Build_Benchmarks=ON` and use the matching runtime
environment.

```bash
cmake --build cbuild --target dense_cpu_benchmark --parallel 6
python3 benchmarks/dense_cpu/run.py --output results/my_machine/cpu/run-01 -- \
  --orders 1,3 --indices 1,8 --samples 3
```

The runner defaults to `cbuild/benchmarks/dense_cpu/dense_cpu_benchmark`; use
`--binary` for another build. Profiles, workload bounds, sample statistics and
comparison rules are described in `dense_common`.

[Workstation evidence](thinkstationP620_RTXA4000/README.md) includes CPU-prefix
controls and matched CPU/GPU fixture runs. Select the CPU rows from mixed-backend
records when comparing host timings.

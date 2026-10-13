# Dense CUDA-small benchmark

This entry point measures the native small-dimension evaluator against CPU and
cuBLAS references using the [shared physical fixture and measurement contract](../dense_common/README.md).
The native target requires `XCA_WITH_CUDA_SMALL`; configure benchmarks with
`Build_Benchmarks=ON` and load the environment matching the configured TRIQS.

```bash
cmake --build cbuild-cuda --target dense_cuda_small_native_benchmark --parallel 6
python3 benchmarks/dense_cuda_small/run.py \
  --candidate-sources c++/triqs_xca/dense/cuda_small \
  --output results/my_machine/native-small/run-01 -- \
  --orders 1,3,6,8 --indices 1,8 --samples 3
```

The runner defaults to `cbuild-cuda/benchmarks/dense_cuda_small/dense_cuda_small_native_benchmark`.
Use `--binary` for another build. `XCA_BATCH_CAPACITY` selects the evaluator batch
capacity, default 128. Backend labels include the capacity, e.g. `cuda_small_b128`.

```bash
python3 benchmarks/dense_cuda_small/run.py --profile throughput --output results/my_machine/native-small/throughput-01
python3 benchmarks/dense_cuda_small/run.py --profile coverage --output results/my_machine/native-small/coverage-01
```

At high orders, select a few topologies/indices to bound runtime. CLI `--help`
lists workload bounds. First-call, construction and warm timings are separate;
all measurements retain matching index lists, numerical checks and provenance.

[Workstation evidence](thinkstationP620_RTXA4000/README.md) documents the measured
small-dimension cases and larger-dimension library controls. GH200-class GPUs are
the target and need separate qualification.

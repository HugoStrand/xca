# Dense CUDA-small benchmark

Bounded physical fixtures measure ordinary/paired self-energy and single-particle
APIs over selected topologies and indices. The shared `harness.hpp` is also used by
CPU benchmarks, evaluator tests and experiments. Its fixture and timing semantics
are unchanged by the directory reorganization.

```bash
cmake --build cbuild-cuda --target dense_cuda_small_native_benchmark --parallel 6
python3 benchmarks/dense_cuda_small/run.py \
  --binary cbuild-cuda/benchmarks/dense_cuda_small/dense_cuda_small_native_benchmark \
  --candidate-sources c++/triqs_xca/dense/cuda_small \
  --output benchmarks/dense_cuda_small/<hardware>/results/native-01 -- \
  --orders 1,3,6,8 --indices 1,8 --samples 3
```

The runner defaults to `dense_cuda_small_benchmark`, which measures the retained
cuBLAS reference using the same fixtures. The native target is available when
`XCA_WITH_CUDA_SMALL` is enabled. Configure with `Build_Benchmarks=ON`, and load
the environment matching the build. The output directory must be new.

Profiles `quick`, `throughput` and `coverage` select bounded workloads; arguments
after `--` override profile defaults. C++ `--help` lists dimension/order/count
bounds. At high orders use a few topologies and indices. The runner forces one
OpenMP/OpenBLAS thread and records raw samples, median/min/max/MAD summaries,
source/binary hashes, commands, environment and library/GPU information.

Wall time includes the complete host-facing call and synchronization. First call
and construction are recorded separately from warm repeated timings. Candidates
are checked against retained CUDA results; low orders additionally enforce CPU
parity. Higher-order CPU differences remain diagnostic. Non-finite values and
shape mismatches fail. Comparisons require identical fixtures and selected indices;
paired seeds and ordinary indices are different work units. Run serially and use
fresh-process repeats before interpreting small gains.

CPU-only measurements now live in [dense_cpu](../dense_cpu/README.md).
[Comparison tools and reports](../dense_comparison/README.md) and
[workstation raw evidence](thinkstationP620_RTXA4000/README.md) are separate.
The target hardware is GH200-class GPUs; qualification on RTX A4000 is local.
Experimental sources and results remain untouched under `experiments/`.

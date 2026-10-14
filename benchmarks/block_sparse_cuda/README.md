# Block-sparse CUDA reference calls

Performance target: GH200-class accelerators. The retained synchronous reference
establishes numerical and complete-call controls; local RTX A4000 measurements do
not establish target dispatch defaults. Existing dense experiments are separate.

With CUDA TRIQS and `Build_Benchmarks=ON`:

```bash
source ~/env/xca-cuda.sh
cmake --build cbuild-cuda --target block_sparse_cuda_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild-cuda/benchmarks/block_sparse_cuda/block_sparse_cuda_benchmark \
  --modes 6 --partition number --observable se --order 3 --count 1 --samples 3
```

Shared workloads, numerical screening, sample definitions and the paired campaign
recorder are in [block_sparse_common](../block_sparse_common/README.md). Results
belong under informative hardware directories here; generic methodology stays here.
No full N=14 atomic model is constructed by this benchmark.

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

`block_sparse_cuda_ordered_benchmark` selects the production ordered operator route
explicitly. It uses the same complete-call fixtures/output as the library frontend
and rejects builds without native sparse kernels. Its backend label is
`cuda_ordered_production`; the private experiment is a separate control. Use this
binary as the CUDA argument to the common recorder, campaign `ordered-production-01`,
with the matched CPU binary. Native compilation is controlled independently with
`XCA_CUDA_BLOCK_SPARSE=AUTO|ON|OFF`; no performance default is implied by compiling it.

`block_sparse_cuda_real_nca_benchmark` opts into the production `RealBatch` NCA
policy and retains ordered operators for fallback. Its complete-call fixture checks
that real occupation-partition NCA self-energy uses the batch route; other orders
and partitions use the reference. Compilation requires native sparse kernels.
The backend label is `cuda_real_nca_batch`. It shares numerical screening and timing
with the other frontends and leaves the reference binary available as a baseline.

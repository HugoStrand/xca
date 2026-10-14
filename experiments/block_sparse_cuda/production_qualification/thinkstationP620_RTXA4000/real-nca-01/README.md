# Real scalar NCA primitive qualification

Focused checks of the internal pointwise batch primitive, before evaluator dispatcher integration. No benchmark or GH200 performance claim.

Build `block_sparse_cuda_real_nca` in each environment, then run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir BUILD -R '^block_sparse/cuda_real_nca$' --output-on-failure
```

CPU metadata/capability, native CUDA reference equivalence, batch tail, cutoff adjacency, live subnormal, invalid metadata recovery, input/output alias rejection, and native-disabled constructor rejection pass. The native memory check passed all three controls with zero errors/leaks before the final capability-only test edits. Its retained log records that exact run.

The native primitive follows sequential field summation and per-output lane addition. The caller must establish exactly real inputs and finite intermediate/output range before using it. These checks do not qualify that future caller or higher-order batching.

The initial final-run CTest spelling using an underscore matched no tests and was corrected to the slash-qualified name above. Only the corrected results are retained here. Source revision and binary hashes are in manifest.json. No cluster time was consumed.

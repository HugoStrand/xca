# E2: CUDA graph replay of the E1 schedule

Bounded prototype: ordinary indexed self-energy, at most128 indices per call.
It preserves E1 at74f2adf and the production baseline064b64e. The evaluator and
owned solver/status code are copied into the graphs namespace; DeviceOps is reused
from the retained serial_stream folder. No production option or source changes.

One graph/executable is cached, keyed by topology and index values, including
caller order and duplicates. G is uploaded on every call; zeroing the accumulated
output remains outside the graph. Graph capture/instantiation happens lazily in the
first call or after a content-key miss. Stable device buffers, immutable operator
pointer tables and distinct solver-status slots remain alive for the graph's whole
lifetime. Capture retains the same arithmetic sequence and host-mode scalar values.
No shared NDA stream/settings are changed. Objects require serialized calls.

`xca.graph.build` marks separately charged capture/instantiation. Replay plus its
completion fence is inside `xca.self_energy.traversal`. Whole API timing includes
upload, cache key construction, graph setup when needed, replay/fence and owning host
result. First/setup costs are reported separately; warm replay never substitutes
for the full API metric. Opaque graph/library allocations are not included in the
application array byte counter; trace-observed peak is reported separately.

```bash
source ~/env/xca-cuda.sh
cmake -S experiments/dense_cuda_small/graphs -B cbuild-experiment-graphs \
  -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Release
cmake --build cbuild-experiment-graphs --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild-experiment-graphs --parallel 1 --output-on-failure
python3 benchmarks/dense_cuda_small/run.py --binary cbuild-experiment-graphs/graphs_benchmark \
  --candidate-sources experiments/dense_cuda_small \
  --output cbuild-small-runs/my-graph-run --profile quick -- --methods se
```

Hash the experiment parent directory to include reused E1 adapters in provenance.
The analytic test adapter enumerates the suites' at-most128 ordinary-SE indices
into an indexed call; unsupported pairs/SPGF remain production CUDA tests. Analytic
one-/two-fermion and complex-rotation tests pass. API tests cover changed G at the
same address, same-address index/topology mutations, cache hit/miss counts, duplicates,
empty vectors, >INT_MAX indices, invalid-prefix recovery, capacity rejection, and
owned-solver independence from shared NDA synchronization. No experimental support
is claimed for unsupported copied API declarations or dynamic interactions.

For profiling, add `--cuda-graph-trace=node` to `nsys profile --trace=cuda,nvtx`.
Default graph-level recording omits node kernels/copies and cannot support the
traversal audit. See RESULTS.md and tracked node-level CSVs; incomplete initial
trace evidence is labeled separately.

Upload-ordering correction discovered during E3: pageable NDA host-to-device copies
use the default stream and can return before DMA completes. The owned nonblocking
stream now waits on an explicit event at constructor/input-upload boundaries.
Low-level DLR/DeviceOps callers must call wait_for_default_uploads() after external
NDA/default-stream uploads before submitting owned-stream reads. The event is
outside captured traversal; arithmetic and precision gates are unchanged. Earlier
measurements are historical and are being revalidated after this correction.
See [CUDA copy semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

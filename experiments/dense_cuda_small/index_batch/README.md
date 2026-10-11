# E3: cross-index batching with shared-LU wide RHS

Ordinary self-energy only. Isolated namespace/standalone build, based on retained
E1 at74f2adf and frozen production064b64e. No reduced precision or explicit inverse.

Physical `(r,B,N,N)` storage is exposed as a contiguous `(r*B,N,N)` device tensor.
N×N products/vertices batch r*B matrices; rank transforms apply the shared r×r
matrix to B*N*N columns, and coefficient conversion uses one cached LU with a
wide RHS. The operator/kernel choices, signs and prefactors remain lane-specific.
Metadata is prepared into immutable device tables before submitting a chunk.
Sequential kernel multiplications and convolution addition order are retained.

The special vertex retains Tkaps reuse per lane. Its ±1 parity multiplication is
separate from hybridization weighting; the final lane results are gathered by pitched
D2D copies into contiguous persistent scratch and accumulated with the original
complex cuBLAS axpy in caller order. No host-side or tree reduction is substituted.

Capacity1..128 is selected through XCA_BATCH_CAPACITY (default8), independently of
caller vector size. Full-capacity tensors have inactive lanes zeroed; tails never
contribute. Calls may span multiple chunks. Each chunk finishes before metadata is
reused, preserving bounded ownership. Orders through12 are supported; full-index
enumeration is bounded to4096, while indexed calls retain long indices/chunking.

One chunk's metadata/schedule is cached by topology/index **contents**, including
order and multiplicity; changing G never invalidates the cache because it is uploaded
every API call. A multi-chunk call may miss repeatedly because ownership is bounded
to one chunk. XCA_BATCH_CACHE=0 disables reuse as a changing-index control. Cache
preparation/misses and chunk fences are part of complete API time, charged to the
first call or subsequent miss. The constructor preallocates384 weight buffers,
64 vertex table triples, scratch and256 distinct solver-status slots. No metadata
is rewritten while queued kernels can read it. Library/opaque handle memory is
excluded from the known array byte counter and needs trace evidence.

```bash
source ~/env/xca-cuda.sh
cmake -S experiments/dense_cuda_small/index_batch -B cbuild-experiment-batch \
  -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Release
cmake --build cbuild-experiment-batch --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild-experiment-batch --parallel 1 --output-on-failure
XCA_BATCH_CAPACITY=128 python3 benchmarks/dense_cuda_small/run.py \
  --binary cbuild-experiment-batch/index_batch_benchmark \
  --candidate-sources experiments/dense_cuda_small \
  --output cbuild-small-runs/my-batch-run --profile throughput -- --methods se
```

Backend names include capacity and the uncached suffix where appropriate. The runner
records these environment settings. Large capacities intentionally do extra work on
small/tail vectors; measure that cost, and use capacity1 as the single-index control.
The benchmark's strict CUDA-reference and low-order CPU gates remain unchanged.
Existing analytic/rotation tests use an ordinary-SE adapter; unsupported methods
remain baseline CUDA checks. Primitive and API tests cover wide solves, rank/edge
operations, tails, duplicates, input changes, content cache invalidation, long indices
and partial invalid prefixes. Correctness suites and memcheck API/primitive runs pass;
measured capacity/cache tradeoffs and repeats are documented in RESULTS.md.

External NDA/default-stream uploads are handed to the owned nonblocking stream with
an explicit event. This is required even for cudaMemcpy without an Async suffix:
pageable uploads can return before device DMA finishes. Primitive users call
wait_for_default_uploads() once after external uploads; the evaluator handles its
own upload/metadata boundaries. A delayed-producer regression exercises this ordering.
See [CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

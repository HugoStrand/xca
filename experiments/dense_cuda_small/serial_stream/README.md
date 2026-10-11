# E1: owned serial CUDA stream

Ordinary self-energy only, using the frozen dense CUDA evaluator at064b64e as
source provenance. These copied files preserve its complex-double arithmetic and
sequential index accumulation. Production sources remain unchanged. Pairs and
correlators are unsupported by the benchmark factory.

The experiment owns a nonblocking stream, cuBLAS/cuSOLVER handles, immutable
per-operand device pointer tables, and65536 persistent managed solver-status slots.
Every queued getrs receives a distinct slot; completion inspects statuses once at
the public return boundary. Exceeding the status capacity fails explicitly; it does
not discard indices. Scratch and model snapshots persist across calls. Host G is
uploaded every call, including when its address stays unchanged. The object is not
thread-safe. No process-wide NDA stream or synchronization settings are changed.

Pointer tables are prepared for all ordinary-SE operator/scratch routes during
construction. Enqueued vertices never overwrite them. The evaluator's return-boundary
fence is inside its traversal NVTX range so timestamp-based GPU audits include all
queued work. Inner phase ranges describe host submission; their timestamps must not
be interpreted as complete asynchronous GPU phase timings.

Build against the existing CUDA baseline tree (no production configure changes):

```bash
source ~/env/xca-cuda.sh
cmake -S experiments/dense_cuda_small/serial_stream -B cbuild-experiment-stream \
  -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Release
cmake --build cbuild-experiment-stream --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild-experiment-stream --parallel 1 --output-on-failure
python3 benchmarks/dense_cuda_small/run.py \
  --binary cbuild-experiment-stream/serial_stream_benchmark \
  --candidate-sources experiments/dense_cuda_small/serial_stream \
  --output cbuild-small-runs/my-stream-run --profile quick -- --methods se
```

`XCA_BASELINE_BUILD` may override the existing baseline build directory. The
standalone opt-in target requires CUDA-enabled TRIQS and the baseline libraries.
CPU-only production builds are unaffected.

The tests reuse existing one-/two-fermion analytic and complex-rotation sources
via a generated test-only adapter. Ordinary SE calls use this experiment; other
methods use production CUDA to keep the existing suites intact. Passing these tests
does not validate experimental pairs/SPGF. `test_dlr.cpp` reuses the existing DLR
reconstruction/reflection/noncommuting-convolution/cancellation tests, adding explicit
stream completion before host assertions. Its original shared-NDA-setting restriction
is inapplicable to the owned solver and is excluded.

See `results/` and RESULTS.md for timing, parity, memory and trace evidence once
collected. Setup, first call and complete API timings follow dense-small-v2; no
prewarming beyond constructor work is hidden from the harness.

Upload-ordering correction discovered during E3: pageable NDA host-to-device copies
use the default stream and can return before DMA completes. The owned nonblocking
stream now waits on an explicit event at constructor/input-upload boundaries.
Low-level DLR/DeviceOps callers must call wait_for_default_uploads() after external
NDA/default-stream uploads before submitting owned-stream reads. The event is
outside captured traversal; arithmetic and precision gates are unchanged. Earlier
measurements are historical and are being revalidated after this correction.
See [CUDA copy semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

# Compact DLR scratch

Performance target: GH200-class hardware. This private E7 control retains production
CUDA arithmetic and solver conventions while investigating buffer liveness. The
production DLR operations and all prior evaluators/experiments remain available.

Completed solve RHS is released before convolution. With a disjoint caller output,
that output stores the diagonal contribution, avoiding an owned diagonal tensor and
its final copy. Exact/partial input overlap retains the original separate diagonal
until input reads finish. All products/additions keep their original order. Shape,
budget and overlap checks precede release/output writes; device errors propagate.
Convolution scratch follows actual rectangles and records whether diagonal is owned.

After a square composed edge, retained scratch falls from six rank tensors to four.
On the next solve, four scratch tensors plus RHS coexist, so the observed method
boundary peak is five, not four. This distinction matters for complete-call admission.
Aliased convolution can retain five owned tensors; subsequent solves can require
six. The full evaluator always uses distinct convolution output, so the private
complete-call control reserves five DLR tiles. Other caller/output/operator reserves
remain conservative; this is not complete allocator/opaque/host memory accounting.

The private source snapshot keeps the public DLR API. Three adapted existing tests
cover cached LU, variable RHS/shapes, reflection, rectangle/mesh convolution, aliases,
cancellation and rejection/recovery. An independent production comparison requires
exact results and checks retained storage and alias-mode switching. Full evaluator
controls reuse eight current observable/API tests. No Python option exposes this
candidate, and no numerical tolerances or public defaults change.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DBuild_Benchmarks=ON -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_compact_dlr_tests \
 block_sparse_compact_evaluator_tests block_sparse_compact_dlr_benchmark \
 block_sparse_compact_evaluator_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild-cuda \
 -R '^experiments/block_sparse_cuda/compact_(dlr|evaluator)$' --output-on-failure
python3 experiments/block_sparse_cuda/compact_dlr/run.py \
 --machine thinkstationP620_RTXA4000 --campaign compact-01
```

Composed-edge timing includes two cached-LU solves, packing, status synchronization,
convolution, allocations/releases and stream completion. Tables, input uploads and
caller buffers are prepared outside timing; downloads/checks are outside timers.
Reference and compact use identical inputs and require exact replay. Report retained
bytes separately from explicit storage observed at method boundaries; neither is a
continuous device allocator peak. Raw samples/provenance belong in hardware folders.
Reallocation cost may dominate small cases, so memory savings alone do not justify a
default change. Compare complete calls before choosing a policy. CPU-only builds omit
these CUDA fixtures; no nvcc/native sparse kernel is required for the DLR primitive.

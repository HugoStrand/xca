# Native rank qualification

Separate experiment for rank26/36 and wider qualification. Configuration copies the
current native evaluator into the build tree under a distinct host namespace. It
changes only the N4/N8 rank eligibility for batching and product kernels. Original
production code and every earlier experiment remain intact. No delegation is used
for ordinary/paired SE or SPGF. `XCA_PROBE_PRODUCTS=OFF` provides library products
with identical pointwise fusion and native batching; ON uses the identical compiled tile4 products.

Use the CUDA environment, configure with clang19 and Release, then build. Root
`cbuild-cuda` must already provide the native static libraries. Snapshot contents
include the generator and benchmark entry. Pass the snapshot directory to the
recorder so actual generated host code is hashed:

```bash
source ~/env/xca-cuda.sh
cmake -S experiments/dense_cuda_small/native_ranks -B cbuild-experiment-native-ranks \
  -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Release
cmake --build cbuild-experiment-native-ranks --parallel 6
python3 benchmarks/dense_cuda_small/run.py \
  --binary cbuild-experiment-native-ranks/native_rank_benchmark \
  --candidate-sources cbuild-experiment-native-ranks/snapshot \
  --output cbuild-small-runs/native-rank26 --profile throughput -- \
  --lambda 40 --dimension 4 --methods se,pairs --indices 8,128,129
```

These are unqualified experiment settings. Every run uses the unchanged numerical
gate, including individual contributions and aggregate results. Promote a policy
only after numerical/physical coverage and independent process/seed repeats.
The local RTX A4000 is a control; GH200-class cards are the optimization target.

# Bounded special-vertex kappa reuse

The performance target is GH200-class hardware; local RTX A4000 measurements are
screening evidence. These private E6 variants retain the complete evaluator API,
production library/native routes, and all earlier experiments.

For a symmetry set of size q, the reference applies each kappa operator to the
same incoming T once per mu, making q*q right applications. A cache with c slots
retains the first min(c,q) products for this path only. Subsequent mu iterations
reuse them while uncached products are recomputed. The right-application count
becomes c + q*(q-c); all q*q weighted additions and q left applications remain.
Singleton sets use the original route, without cache allocation. Two-slot and
sixteen-slot variants show partial and full retention for the bounded fixtures;
16 slots is a finite cap, not a claim to handle arbitrarily large sets resident.

No sum is factored or reassociated. Mu then kappa traversal, reflected coefficient
indices, parity, weighting and accumulation order remain the reference's. Cached
products are pure functions of the unchanged T and immutable model. Their ownership
ends with the special vertex, so changed G, topology, coefficients or masks cannot
reuse old numerical values. Device operations remain synchronous.

The explicit admission bound adds min(cap,largest nonsingleton set) maximum-sized
rank buffers before input upload. Products use actual rectangular extents; this is
a conservative worst-case envelope, not a measured total host/opaque allocator
peak. Correlator admission inherits this extra conservatism although correlators
do not use this cache. Increasing the cap may reject workloads that the reference
accepts. No production fallback or dispatch policy exposes these variants.

Both variant test executables reuse the eight evaluator physics/API tests. Two
extra cases compare against the retained GPU reference bit for bit with both sparse
backends, a four-field number-symmetry set, rectangular paths, changed complex G,
zero masks and multiple indices; and test exact extra payload accounting, rejection
before submission, empty ranges and recovery. This experiment changes operator
reuse, not the DLR algorithm. The original CPU evaluator already retains all kappa
products, so claims of hardware speedup must use a suitable matched DLR CPU control.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_kappa_2_tests \
 block_sparse_kappa_16_tests block_sparse_kappa_2_benchmark \
 block_sparse_kappa_16_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild-cuda \
 -R '^experiments/block_sparse_cuda/kappa_(2|16)$' --output-on-failure
```

Run complete-call comparisons with `run.py`, using the retained ordered production
benchmark and the two private binaries. Setup and original-CPU control time are
separate from call timing; source/binary hashes, raw samples, errors, discovery
counts, storage bounds and commands are recorded in a new hardware campaign folder.
All variants require deterministic replay. Profiles and verification must finish
before timing. Primitive operation-count savings do not establish full Solver or
GH200 gains. Wider sets, large-sector streaming, optimized bosonic mixing and
correlator scheduling remain separate gates.

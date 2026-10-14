# Mapped kappa weighting

The target is GH200-class hardware. This private E6/E7 candidate retains the
complete CUDA evaluator API and all reference/cache controls. It optimizes only
multi-field special vertices whose kappa blocks have at most one nonzero per
column; general operators and singleton sets use the unchanged reference method.

An owned column table stores a source index and complex value for each field and
output column, with -1 for empty columns. Each GPU thread forms one mu element by
visiting fields in their original order: right sparse product, rank-dependent
weight, sequential addition. The original left sparse application and mu-to-output
addition remain separate. No algebraic factoring, tree reduction or atomics occur.
Complex arithmetic uses explicit products with contraction disabled. Fusion may
change rounding relative to library primitives; qualification uses the existing
strict numerical controls rather than promising universal bitwise equivalence.

Maps and weights are rebuilt from the immutable model/path for each special vertex;
no numerical data are cached across calls. Their ownership lasts through stream
completion, including default-stream upload handoff. Input/output shapes are checked.
Both structural classification and all transfers, allocations, weights, kernels,
synchronization and result construction enter complete-call timing.

Admission adds q*(Dmax*(sizeof(long)+sizeof(complex))+r*sizeof(complex)), where q is
the largest nonsingleton set. Actual maps use rectangular block dimensions. This
conservative allowance is also charged for unsupported paths and correlators that
do not use the mapper. Existing function/rank-buffer envelopes are retained; this
is not complete host/opaque allocator peak admission or a claim that N=14 fits.

The CPU-testable classifier checks CSR structure without dropping tiny values.
Tests cover empty columns/payloads, complex entries, incompatible shapes and
multiple entries per column. The native evaluator suite reuses eight physics/API
cases and adds seven controls: rectangular changed-input/mask comparisons, complex
bosonic mixing/reflection, pre-submission budget rejection and empty ranges,
non-mapping operators on accepted live paths, a rectangular complex kernel control,
cancellation-sensitive field order and a nine-point NCA cutoff neighborhood.
Higher-order cutoff-adjacent cases remain a separate gate. Original tests/tolerances
and production code are unchanged.

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DBuild_Benchmarks=ON -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_mapped_special_tests \
 block_sparse_column_maps_tests block_sparse_mapped_special_benchmark --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ctest --test-dir cbuild-cuda \
 -R '^experiments/block_sparse_cuda/(column_maps|mapped_special)$' --output-on-failure
python3 experiments/block_sparse_cuda/special_vertex/run.py --with-mapped \
 --machine thinkstationP620_RTXA4000 --campaign kappa-mapped-01
```

CUDA compilation is optional: CPU-only/native-compiler-disabled builds retain the
host classifier tests and omit the fused evaluator/kernel targets. GPU comparison
frontends explicitly require production ordered sparse support. The shared recorder
retains raw GPU results, hashes and commands in hardware benchmark folders and
compares both bounded caches with the reference and mapper. Finish tests/profilers
before isolated timings. Selected-index results do not establish Solver/topology
throughput or target dispatch. Next replay actual fourteen-mode sparse maps on
bounded row slabs without atom_diag, and continue large-sector streaming/admission.

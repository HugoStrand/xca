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

## Actual Fock-sector row slices

`block_sparse_mapped_slab_benchmark` generates all canonical fields between adjacent
half-filled number sectors, using occupation masks and exact fermionic signs.
Creation uses their adjoints. No Hamiltonian, atom_diag diagonalization or full
solver evaluation is constructed. CPU checks include known four-mode matrix entries
and fourteen-mode number/complement completeness; each field has 1716 entries in
3003x3432 or 3432x3003 blocks.

The primitive forms one mu row slice, with deterministic dense complex input and
complex rank-dependent weights. Rows are bounded to 128 and the source-sector
size, rank to 26, explicit tensor/map/operator payload to one GiB with free-memory
headroom. A matched serial CPU computes the same ordered field sum. GPU controls
separate right applications/weights/additions with all operator snapshots resident,
which favors that reference versus rebuilding snapshots in the production loop.
The resident mapper times synchronized execution with fixed inputs/maps/weights.
A separate route includes map rebuilding, uploads and mu download while input and
output tensors remain allocated. It does not include input upload, the left operator,
remaining mu summation, path cutoff, DLR transforms, full output, MPI or model setup.

Setup and controls are outside timing; resident GPU result downloads are outside
those resident timers. CPU and maps/download routes return owned host results inside
their timers. Validation includes finite outputs, strict 1e-13 relative control and
exact resident mapped replay. Profiler/sanitizer times are excluded from comparisons.

```bash
python3 experiments/block_sparse_cuda/mapped_special/run_slabs.py \
 --machine thinkstationP620_RTXA4000 --campaign fock-slabs-01
```

The recorder runs both directions for modes 4, 10 and 14, with actual sector widths,
including rank 26 and 32/128-row fourteen-mode slices. Three timed repetitions per
route/case remain one process, not independent trials. Raw rows and provenance go
under this experiment's hardware folder. These replays qualify map indexing at
large dimensions and primitive memory costs, not fourteen-mode full-call admission.
GH200 remains the target; serial CPU/compiler optimization and transfer costs must
remain visible when assessing any primitive advantage.

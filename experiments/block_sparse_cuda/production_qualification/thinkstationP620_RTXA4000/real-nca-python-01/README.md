# Real NCA Python and MPI qualification

Code fbbde67 adds consolidated `nca_backend='real_batch'`, build capability query,
route observer, and Solver `block_sparse_nca_backend` preferences. Default is
reference; archive/deepcopy retains preferences and older archives restore reference.
CPU capability queries return false without loading the CUDA module. Library-only
CUDA rejects explicit native NCA selection.

Nine native methods pass (29.44 s CTest). CPU archive/default/capability passes (.40 s,
eight GPU methods skipped). Library-only CUDA passes eight methods (13.08 s, one
native-only test skipped). Two MPI ranks each pass all nine methods in 134.039 s;
launcher exit zero. Rank-local invalid selection and budget failure are collective,
and later valid selection/evaluation recovers. Real dynamic NCA, changed complex
input, larger blocks, higher orders, SPGF/correlators, host placement and solver
self-consistency retain reference agreement. Raw unittest output is included.

Focused serial tests use the normal environment switchers, OMP/OpenBLAS threads one,
and CTest `^Py_block_sparse_cuda$` in the three build trees. MPI launches the selected
CTest command under `mpirun -n 2` with its JSON environment modifications applied.
All GPU stages are serial; the MPI stage intentionally has two ranks sharing the
local device. Known X11 authorization startup diagnostics accompany a successful run.
This is not an automatic approval rejection.

Final committed C++ source builds after whitespace-only formatting; capability
imports and module hashes are recorded. Feature tests ran before that formatting,
with identical runtime logic. Initial standalone capability probes accidentally
replaced the TRIQS Python path and failed import; corrected probes prepend build and
dependency paths while preserving the switched environment. Only corrected capability
logs are retained. No new broad-suite or target-hardware claim, cluster allocation,
install or push. RTX A4000 evidence is a development proxy for GH200-class targets.

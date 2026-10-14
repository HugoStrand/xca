# Production backend qualification

Build all configured targets with binding regeneration disabled, then run production
CTest with `-E '^experiments/'`. Experimental targets are built but their tests are
excluded. CPU-only disabled GPU entries do not qualify device execution. Verify
live imported capability flags alongside configured support and actual test output.

Use separate CPU/CUDA build trees and environment switchers. Library-only CUDA sets
XCA_CUDA_SMALL=OFF and XCA_CUDA_BLOCK_SPARSE=OFF against CUDA TRIQS. Serialize GPU
CTest and MPI runs. No install/dependency refresh/shared-checkout changes required.

For integration use build-tree Python and cached pyed/adapol paths, single-thread
BLAS/OpenMP and `mpirun -n 2 python3 test/python/block_sparse_cuda.py`. Inspect both
rank results and launcher exit code. The suite covers observables/dynamic channels,
archives, capabilities, host placement, topology bounds and collective recovery.

Hardware folders retain manifests, exact test lists, CTest summaries and focused
Python/MPI transcripts. Full temporary logs are hashed and may disappear. Tests
are correctness evidence, not performance or full fourteen-mode qualification.

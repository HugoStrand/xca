# Production suite refresh after storage integration

Source code 85ecb2d (stream fix 7054f0c); checkpoint 881667a.
All configured targets build in CPU, native CUDA and library-only CUDA trees,
including retained experimental targets. No code changed during qualification.

| Tree | CTests | Result | Seconds |
| --- | ---: | --- | ---: |
| CPU | 85 | all pass | 58.90 |
| Native CUDA | 94 | all pass | 389.90 |
| Library-only CUDA | 85 | all pass | 239.64 |

CPU runs two CTests concurrently. CUDA runs one CTest at a time, and the two
CUDA suites run sequentially. OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1.
CUDA test selection excludes ^experiments/; the CPU run includes its host-only
column-map experiment. Exact names and configured flags are in test-inventory.json.
Library-only has XCA_CUDA_SMALL=OFF and XCA_CUDA_BLOCK_SPARSE=OFF, but uses CUDA TRIQS.
CPU disabled GPU entries and native-only skips are not device qualification.

Full raw LastTest transcripts are retained for CPU and native CUDA. A concurrent
CTest --show-only inventory query overwrote the library-only LastTest log with an
empty inventory transcript; that file is retained and is not raw test evidence.
The complete library-only CTest summary records all 85 results and exit success.
Earlier focused raw library output is retained in sibling storage-python-01.
No test rerun is inferred from the inventory query.

The sibling storage-python-01 retains the two-rank MPI qualification and diagnosed
failures. This refresh is correctness evidence on RTX A4000, not performance or
GH200 Python/MPI qualification. No Arrhenius job was submitted. Tracked target
usage remains 1704 GPU-seconds (0.47333 nominal GPU-hours), within a shared monthly
100 GPU-hour ceiling. Full physical N14 and composed large-Solver capacity remain
unqualified. Original log hashes are recorded before archive whitespace normalization.

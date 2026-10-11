# Custom cached-LU solve: numerically rejected

The trace after E4 puts most remaining kernel time in solves. This follow-up uses
one CUDA thread per RHS with cached factors/pivots: U^T forward substitution,
unit-diagonal L^T backward substitution and reverse pivots. It eliminates layout
packing and retains complex double, no inverse/refactorization/conjugation.
Rank<=40 bounds private scratch; compiler reports47 registers and640B stack/thread,
zero register-spill loads/stores. XCA_SOLVE_MODE=0 is the unchanged library control.

21 alternating warmed primitive calls at B128 give:

|Rank |Library median |Custom median |Custom coefficient relative difference |Custom reconstruction residual |
|---|---:|---:|---:|---:|
|16 |.320ms |.104ms |2.84e-15 |7.08e-17 |
|26 |.458ms |.264ms |1.21e-4 |7.24e-17 |
|36 |.634ms |.481ms |7.39e-5 |8.33e-17 |

Library reconstruction residuals are5.59–6.21e-17. These are primitive host timings
including completion, without transfers, not accepted full-API speedups. Only one
process is timed because the numerical rejection prevents promotion. Commands,
source/binary hashes and raw CSV are retained. micro-final.csv measures both residuals;
earlier micro.csv used zero as an unmeasured library reconstruction placeholder.

Strict CTest rejects custom mode: wide coefficient primitives and public API fail.
The ordinary fixture order12 long-index result has norm2.69e-31, error3.09e-31,
relative1.147; a dense complex basis case reaches relative errors above1. No tolerance
was changed. The benchmark rejection reproduces at norm6.61e-18, error7.66e-28,
relative1.158e-10 (just over the existing threshold), with a nonzero exit and manifest.
Library-control mode0 passes all six CTest entries. CUDA memcheck of all three-rank
primitive cases reports0 errors, supporting a rounding/conditioning explanation
rather than invalid accesses. Full-CUDA-reference parity remains the acceptance gate.

Sequential substitution changes arithmetic relative to blocked library TRSM;
tiny residuals do not imply sufficiently close coefficients or high-order diagrams.
Retain cuSOLVER/cuBLAS LU solves in fused_small. Matching blocked arithmetic or
separate DLR-stability work may justify a later attempt; this probe is not promoted.

Build with the CUDA environment, clang++-19 and the standalone CMake directory:

```bash
source ~/env/xca-cuda.sh
cmake -S experiments/dense_cuda_small/fused_solve -B cbuild-experiment-solve \
  -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Release
cmake --build cbuild-experiment-solve --parallel 6
XCA_FUSED_MODE=2 XCA_SOLVE_MODE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild-experiment-solve --parallel 1 --output-on-failure
XCA_FUSED_MODE=2 XCA_SOLVE_MODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild-experiment-solve/solve_micro
```

Custom mode1 CTests intentionally expose the rejection; they are not marked WILL_FAIL.
No production files, benchmark tolerances or accepted E4 solver paths were changed.

# Time-ordered DLR convolution

This standalone CPU benchmark compares three implementations: the original
cppdlr algorithm reproduced in a benchmark-only helper, the fixed cppdlr
convolution, and XCA's shared `cppdlr::fast_convolve`. All timings include
conversion of two complex arrays from imaginary-time values to DLR
coefficients, the convolution, and writing its values to an output array.
The installed reference must include the cppdlr fix that hoists the two
Hilbert transforms out of the orbital-product loop (`prq_convolve_opt`, commit
`b9b24ab` in the local cppdlr checkout).

This measures one numerical kernel. It does not measure a complete diagram
evaluator or claim an end-to-end solver speedup.

## Original algorithm reproduction

`legacy_convolve.hpp` reproduces the pre-fix rank-three, time-ordered
convolution: each of the `r` DLR-node iterations recomputes the two complete
Hilbert transforms before taking their node slices. For square matrices of
dimension `D`, this costs `O(r^3 D^2 + r D^3)`, compared with
`O(r^2 D^2 + r D^3)` after the fix. The extra power of `r` applies to the
Hilbert-transform work, not to every part of the convolution.

The reproduction uses the installed cppdlr coefficient conversion and array
operations. Its real convolution matrices are prepared once outside the
timed calls. It does not change the installed library. Its output and the fast
helper's output are checked against the fixed cppdlr reference.

## Shared API

The implementation is in `c++/cppdlr/fast_convolve.hpp`. Dense, batched dense,
FewOrb and block-sparse diagram evaluation share its DLR arithmetic. For operands
shaped `(r,m,n)` and `(r,n,p)`:

```cpp
cppdlr::fast_convolve convolution(itops);
cppdlr::fast_convolve_workspace workspace(r, m, n, p);
convolution.convolve(beta, f_values, g_values, out_values, workspace);
```

The helper borrows an unchanged `imtime_ops` that must outlive it. It prepares
three real DLR matrices once. Five scratch arrays belong to the workspace;
each concurrent call needs its own workspace and output. The helper supports
time-ordered convolution of complex-double, contiguous C-layout arrays,
including rectangular matrix operands. A separate overload accepts independent
convolutions in packed `(rank,batch,rows,columns)` order with a
`fast_convolve_batched_workspace`. A smaller active shape can use existing
workspace capacity through packed views.

Values are copied into coefficient buffers to preserve the inputs. Two
right-sided triangular solves use the existing LU factors directly in those
buffers, followed by reverse pivot swaps. Both solves and DLR-axis transforms
use real BLAS on the interleaved real and imaginary components of the complex
arrays. They need no separate real/imaginary arrays or Fortran-layout buffer.
Spatial matrix products remain complex. The convolution uses output-parameter
GEMMs and reuses scratch without creating owning NDA temporaries. Output may
overwrite either input; workspace buffers must remain separate. This arithmetic
optimization does not depend on small matrix dimensions.

The triangular solves call the raw `F77_dtrsm` wrapper and link NDA's exported
`nda::blas_lapack` target. The helper does not modify NDA or explicitly invert
the interpolation matrix. FewOrb additionally uses the shared coefficient and
transform operations to retain selected edge operands; that cache is separate
from the convolution helper and is not exercised by this benchmark.

## Build and run

Use an installed CPU TRIQS with the fixed cppdlr routine. On the development
workstation:

```bash
source ~/env/xca-cpu.sh
cmake -S benchmarks/fast_convolve -B cbuild-fast-convolve \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-19
cmake --build cbuild-fast-convolve --parallel 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild-fast-convolve/fast_convolve_benchmark
```

An optional argument sets an upper bound on the matrix dimension, between `4`
and `1024` (default `512`). The benchmark visits powers of four starting at
`4`: the default therefore measures `4`, `16`, `64`, and `256`. An upper bound
of `1024` also considers that dimension, subject to the guards below. The
benchmark has its own CMake project and does not change XCA's main
configuration or install any files.

## Measurement

The basis uses `lambda=100000` and tolerance `1e-12`; the actual DLR rank is printed.
Each input is a deterministic sum of three smooth exponentials multiplying
complex, noncommuting dense matrices. The matrix coefficients scale as
`1/sqrt(D)`. The inverse temperature is seven.

Basis construction, convolution helper construction, input generation and fast
workspace allocation are outside timed calls. The reference's lazy convolution
setup is warmed before timing. Outputs are compared before timing, and the
maximum absolute discrepancy divided by the maximum reference magnitude is
reported. A discrepancy above `2e-10` fails the benchmark.

Each implementation is warmed and timed in five short batches, reporting the
median time per call. Batch length adapts toward 20 ms. Calls slower than 100 ms
use three samples; calls slower than 500 ms use only the measured warm-up call.
The output includes repetitions and sample counts so these cases are visible.
A consumed checksum prevents unused-result elimination.

Before allocating the `D=1024` inputs, the benchmark projects the previous
case's slower timing using cubic dimension scaling. It skips the case if this
projection exceeds three seconds per call or a conservative peak-storage
estimate exceeds 2 GiB. The memory estimate allows 24 simultaneous full-sized
complex arrays; it is a guard, not a measured memory high-water mark. Cubic
projection is also a guard, not a performance prediction: the DLR-axis
transforms scale quadratically with matrix dimension. Skipped rows contain no
measured timings.

The original algorithm has an additional runtime guard: at `D >= 256`, its
previous measured time is scaled by the square of the dimension ratio. If that
projection exceeds three seconds, the original-algorithm case is skipped
before its first call; the fixed and fast paths can still run. This projection
accounts for the dominant repeated transforms but omits cubic matrix-product
growth, so it is a screening estimate rather than a measured timing or a hard
runtime bound. CSV fields record the original algorithm's status, time,
output discrepancy, repetitions, samples, and skip reason separately.

Output is CSV with metadata lines prefixed by `#`. Store machine-specific
results and their interpretation in a hardware-named subdirectory.

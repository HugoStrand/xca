# Block-sparse temporary storage and convolution

This CPU experiment compares ways to execute the same prepared backbone segment
while changing temporary storage, vertex multiplication, and DLR convolution.
It is isolated from the production evaluators and is excluded from the default
build. The existing fast-convolution implementation remains a draft; this
experiment does not integrate it into `Action` or either evaluator.

The reference cppdlr installation must contain the fix that computes the two
Hilbert transforms once per convolution rather than once per DLR node
(`prq_convolve_opt`, commit `b9b24ab` in the development checkout). The older
algorithm is not part of this comparison.

## Variants

All variants reserve two arrays of capacity `r * Nmax * Nmax` for the running
result and the edge buffer. They execute identical operator products, vertex
kernels and time-ordered convolutions on the active dimensions.

| CSV variant | Active storage | Vertex multiplication | Convolution |
| --- | --- | --- | --- |
| `padded_matmul` | Slices of `(r,Nmax,Nmax)` | Current `nda::linalg::matmul`, assigned back to the same slice | Fixed cppdlr |
| `padded_gemm` | Slices of `(r,Nmax,Nmax)` | Output-parameter GEMM into the second buffer, followed by a storage swap | Fixed cppdlr |
| `packed_gemm` | Contiguous `(r,m,n)` views into the reserved storage | Same output-parameter GEMM and swap | Fixed cppdlr |
| `packed_fast` | Contiguous `(r,m,n)` views into the reserved storage | Same output-parameter GEMM and swap | Capacity-adapted fast-convolution draft |
| `padded_matmul_active` | Same as `padded_matmul` | Same as `padded_matmul` | Fixed cppdlr |

The fifth variant is a control for final scaling. Production correlator
evaluation scales only the active forward result and does not scale the backward
half. All physical fixtures follow that behavior, so the fifth variant is
identical to the first for those fixtures. One explicitly marked synthetic
fixture models the full-buffer scaling currently used by self-energy
evaluation: `padded_matmul` scales the complete reserved array, while the other
variants scale only the active result. Its CSV field `baseline_full_scale` is
one. This avoids attributing the self-energy-specific padding cost to every
block-sparse computation.

The padded matrix slices have unit column stride and a larger row leading
dimension. Direct GEMM can consume them without packing. The current NDA
`matmul` instead creates a result matrix and packs noncontiguous inputs. The
packed variants also make the complete time-dependent active array contiguous,
so it can be reshaped directly into an `r`-by-`m*n` matrix for DLR operations.

After a dimension-changing vertex, the GEMM variants swap ownership of the
running-result and edge buffers. Existing views are discarded and new views
use the new dimensions. The subsequent edge preparation overwrites the unused
buffer. This preserves the inputs without adding another full-capacity buffer
or performing an overlapping packed matrix update.

`CapacityConvolve` in `kernels.hpp` is an experiment-local adaptation of
`c++/cppdlr/fast_convolve.hpp`. It retains the same LU-based coefficient
conversion and convolution formula, but uses five reusable arrays of capacity
`r * Nmax * Nmax`. Each call creates non-owning views with the current operand
shapes. There is no per-shape cache, resizing, or NDA array allocation inside
that convolution call. BLAS may use internal storage. The fixed cppdlr and
historical `matmul` variants retain their ordinary temporary allocations.

## Fixtures and scope

The basis uses `beta=2`, `lambda=100000` and tolerance `1e-12`. The executable
prints the actual DLR rank rather than assuming it is exactly 100.

Physical fixtures use the C++ evaluator-test models and production Action,
Backbone and block-path construction:

| Model | Block structure exercised |
| --- | --- |
| `density4` | Four fermion modes with diagonal density interactions and one-dimensional sectors |
| `unequal_sets_nA0` | Unequal symmetry sets and a retarded density interaction |
| `n_partition_spin_xyz` | Total-number symmetry, block dimensions `1,4,6,4,1`, and retarded spin operators |

For each model, the benchmark samples 128 deterministic indexed assignments
at order three and topology `{{0,3},{1,4},{2,5}}`. From the discovered paths it
retains a forward and a backward correlator segment, preferring paths with
dimension changes and then larger blocks. These are selected numerical
segments, not a representative average over all assignments or paths.
The selected operators, propagator blocks, edge kernels and vertex kernels
are materialized before timing.

Synthetic fixtures provide controlled shape coverage in both directions:
uniform dimensions one and four, growth and shrinkage through dimensions
`1,4,16`, and paths reaching dimension 64. The mixed cases reserve capacity
for dimension 64 even when the chosen path uses only smaller blocks. Their
smooth complex propagators and rectangular operators are deterministic, but
they are not presented as a physical symmetry decomposition. The explicit
full-buffer-scaling control is forward only.

Every fixture has two timing phases:

- `segment`: copy the initial active data, multiply each vertex, apply its
  kernel when present, prepare and convolve the following edge, and apply
  the forward prefactor.
- `vertices`: the same operations with every edge preparation and convolution
  omitted. This isolates the storage and vertex overhead; it is not a complete
  diagram contribution.

Forward steps compute `F * T` followed by `edge * T` convolution; backward
steps compute `T * F` followed by `T * edge` convolution. Vertex-kernel
reflection is already represented in the supplied backward kernel values.

Both phases exclude model construction, path discovery, operator lookup,
edge-kernel preparation, endpoint sums, correlator joins, final time reflection,
full-evaluator reset and output accumulation. Buffer and convolution-table
construction are also outside timing. These results therefore measure prepared
numerical work, not full-evaluator or solver speedups. Allocation counts are
not instrumented.

## Build and run

Use a configured Release CPU build with `Build_Tests=ON` and the fixed cppdlr
installation on its dependency path. The benchmark links XCA's test-fixture
library; it is not a standalone CMake project or a timing-based CTest.

```bash
# Activate the CPU TRIQS/XCA environment used for the existing build first.
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF
cmake --build cbuild --target block_sparse_layout --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/block_sparse_layout/block_sparse_layout > layout.csv
```

An optional argument filters case names by substring, for example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/block_sparse_layout/block_sparse_layout mixed_small
```

Fixture preparation and the convolution preflight still run before filtering.
CSV metadata lines begin with `#`. Store hardware, software provenance, raw
results and interpretation in a machine-specific subdirectory.

## Validation and timing protocol

Before replay timings, the capacity-adapted convolution is checked against the
original fast-convolution draft using square and rectangular products,
changing dimensions in the same workspace, and overwriting either input when
shapes permit. Every replay variant is then compared with `padded_matmul`
before and after repeated calls. This also exercises buffer-role swaps and
reuse of scratch contents. Zero or nonfinite reference fixtures are rejected.

Each output and reference element is checked for finite real and imaginary
parts. The accepted maximum absolute discrepancy is
`2e-10 * max_abs(reference)`. CSV `relative_max_error` reports the maximum
absolute discrepancy divided by the reference's maximum absolute magnitude.

Each variant is warmed before timing. A short calibration chooses between
one and 1000 repetitions per trial, targeting about 8 ms. Normally five trials
are taken; if any variant's calibrated call exceeds 100 ms, the phase uses
three trials. Variant order rotates between trials. The CSV reports median,
minimum and maximum microseconds per call, repetitions, trial count and
speedup relative to `padded_matmul`. A consumed checksum prevents unused-result
elimination. Small differences should be judged against the observed spread,
not just the median ratio.

## Possible integration boundary

If adopted, the immutable fast-convolution helper and its DLR tables can belong
to `Action`, constructed from the mesh's `imtime_ops`. Mutable scratch belongs
to each evaluator or concurrent call. `block_dims` supplies a conservative
maximum capacity for sparse evaluation; dense evaluation is the fixed-dimension
case of the same workspace interface. The experiment combines helper and
scratch in `CapacityConvolve` only to keep this comparison isolated. It does
not prescribe a new production class hierarchy or a cache of shape-specific
workspaces.

These are CPU experiments. Future GPU design targets high-end hardware such
as GH200; neither a local CPU timing nor the workstation's RTX A4000 establishes
performance on those targets.

# Complete block-sparse prepared-backbone calls

This CPU benchmark measures the public prepared-backbone evaluator calls,
including sector-path discovery, endpoint sums or correlator joins, convolution,
reflection, temporary/result allocation, and the evaluator's current reset work.
Backbone preparation, model construction, baseline serialization, and result
validation are outside the timed region. It complements the numerical-segment
replays in `../block_sparse_layout`; it does not measure the Python solver or
complete topology sums.

The three physical models reuse the evaluator-test fixtures:

- `density4`: four fermion levels with density-density interactions and automatic
  partitioning; all sectors have dimension one.
- `unequal_sets_nA0`: mixed sectors up to dimension two, with a dynamical density
  interaction and unequal operator symmetry sets.
- `n_partition_spin_xyz`: particle-number sectors up to dimension six, with spin
  operators including complex matrix elements and off-diagonal interaction
  coefficients.

Each model is run at its original test mesh (rank 13) and at cutoff `1e5`,
tolerance `1e-12` (rank 103), both at beta two. The rank-103 propagator is rebuilt
from the same atomic Hamiltonian on that mesh and receives the same smooth,
complex, noncommuting dressing as the test fixture. Maximally crossing topologies
at orders two, three and five exercise self-energy and single-particle correlator
calls with the numerical operator lists prepared once.

## Selection and measurement

A deterministic search (seed derived from 1789, order and diagram kind) inspects
at most 512 distinct backbones per case. It retains up to eight nonzero productive
backbones and up to eight with no completed supported path. Productive results
must have maximum magnitude above `1e-24`; zero-path outputs must be exactly zero.
Selection stops when both samples are full. Actual sample counts and the searched
candidate count are reported, so this bounded sample is not presented as a
representative distribution over the full expansion. Missing zero-path samples
are reported on stderr.

Path counts include nonempty endpoint selections and, for self-energy, closing
sector paths. A zero-completion correlator can still evaluate one or both
backbone segments before discovering that no join contributes; its segment count
is reported separately. Thus the zero-path timings include the current
orchestration cost rather than assuming all zero cases return immediately.

Each timed pass visits the selected backbones individually and consumes one
output element. Three trials report median/minimum/maximum microseconds per
backbone. A warm-up pass chooses one to five repetitions per trial, targeting
about three milliseconds when calls are short. Returned outputs are checked
before and after repeated reuse. No numerical-library threads or MPI ranks are
created by the benchmark.

## Build and preserve a baseline

From the repository root, using the existing CPU build:

```bash
source ~/env/xca-cpu.sh
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF
cmake --build cbuild --target block_sparse_convolution --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/block_sparse_convolution/block_sparse_convolution \
  record /tmp/xca_sparse_before > /tmp/xca_sparse_before.csv
```

Record mode exclusively creates `/tmp/xca_sparse_before.h5`; it refuses to replace
an existing reference. For every selected backbone it saves the exact ordinal,
sector dimensions, and every complex output element. Self-energy arrays are
flattened in sector order, then in each array's native iteration order. The
companion `/tmp/xca_sparse_before.indices.csv` records ordinals, path/segment
counts, and output magnitudes in a compact reviewable form.

After changing the implementation and rebuilding, compare against that same
reference:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/block_sparse_convolution/block_sparse_convolution \
  compare /tmp/xca_sparse_before > /tmp/xca_sparse_after.csv
```

Compare mode reuses the recorded selections; it does not resample according to
the new implementation. Every output element is compared with its recorded
reference, with tolerance `2e-10 * max(abs(reference))` for that backbone and no
absolute floor. Zero references must stay exactly zero. This is a benchmark
validation bound; the stricter analytical and evaluator tests remain separate.

An optional final argument filters case labels by substring, for example
`n_partition_spin_xyz_r103_order5`. Use the same filter when recording and
comparing a partial archive. Keep large numerical archives in `/tmp` or another
local artifact directory; store compact timing/ordinal CSVs, source revisions,
and hardware information in a machine-specific subdirectory when publishing
results. Temporary archives must be retained or copied elsewhere if the baseline
is needed after a machine cleanup.

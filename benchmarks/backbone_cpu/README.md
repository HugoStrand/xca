# CPU diagram preparation costs

A small diagnostic for the preparation/execution boundaries introduced by
`plan_backbone.md`. It measures shared assignment preparation, block-sparse path
planning and operand binding, complete self-energy calls, and single-particle
correlator range calls. It uses the C++ test models and a fixed, bounded sample
of indexed assignments. It does not compare against the pre-refactoring code or
measure solver convergence, MPI throughput, or GPU performance.

## Build and run

With a configured CPU build and `Build_Tests=ON`:

```bash
cmake --build cbuild --target backbone_cpu --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/backbone_cpu/backbone_cpu unequal_sets_nA0 3 8
```

Arguments are model, expansion order, and repetitions per timing trial. The
executable is excluded from the default build and is not a timing-based CTest.
It links the existing test fixture library. Source the appropriate TRIQS/xca
environment before building or running it.

Useful models:

| Model | Purpose |
| --- | --- |
| `density4` | Four fermionic states with diagonal density interactions; 16 one-dimensional sectors |
| `unequal_sets_nA0` | Unequal fermionic symmetry sets and a retarded density interaction |
| `n_partition_spin_xyz` | Total-number symmetry with dimensions 1,4,6,4,1 and coupled retarded spin operators |

Other model names from `test/c++/test_utils/evaluators.hpp` are accepted. The
single topology joins vertex `a` to `a + order`. At most 64 stratified assignments
are sampled across the full flat-index space, using RNG seed 1789. The output
includes the total assignment count and the number of discovered paths in this
sample. Samples with few surviving paths measure pruning and call overhead,
not the cost of a representative nonzero contribution.

## Interpreting the output

Times are microseconds per sampled assignment, with median/minimum/maximum of
five trials after warm-up. Setup, operator materialization, support construction,
and model diagonalization are outside the timed regions.

- `flat_preparation` includes decoding and shared recipe preparation;
  `logical_preparation` starts from already decoded indices.
- `sign_and_kernel_binding` starts from prepared assignments.
- `path_discovery` counts candidates without matrix arithmetic.
- `paths_and_operand_binding` includes sign preparation, discovery, operand
  binding, and the diagonal-output closure check. It overlaps the preceding
  phases; these rows must not be added together as a profiler breakdown.
- `prepared_self_energy_call` includes path planning, binding, numerical work,
  result packing, and reset, using a native block propagator. The flat call also
  constructs the topology and prepares the assignment. Differences between these
  two measurements can be smaller than timing noise.
- `single_particle_range_call` also includes public propagator conversion and
  observable setup, then fragment planning, execution, and joins.

All phases consume a checksum. After warming both public range paths, the program
samples Linux resident memory before and after repeated singleton range calls.
It reports calls **per observable kind**; both self-energy and single-particle
functions run that many times. RSS is `-1` on platforms without `/proc/self/status`.
RSS includes input/model storage, the capped sample, allocator caches, and the
whole process; it is not an allocation counter or a proof against every leak.

The implementation's memory bound should also be inspected: one assignment and
one self-energy path workspace are streamed, while correlator fragments are
retained only for the sectors of the current assignment. Neither path stores all
assignments or all possible self-energy paths. The benchmark itself retains at
most 64 recipes for component measurements.

Machine-specific data and interpretation belong in subdirectories. These CPU
measurements guide later work; future GPU optimization targets high-end hardware
such as GH200, and cannot be predicted from these CPU timings or the local RTX
A4000.

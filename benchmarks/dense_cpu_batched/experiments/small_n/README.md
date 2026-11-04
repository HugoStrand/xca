# Small dense CPU experiments

These isolated probes investigate why grouping N=4 backbones gives modest CPU
speedups, and which arithmetic or reuse changes are more promising. They do not
change the production evaluator or convolution. N=4 means two fermion modes.

Build with the same settings and dependencies as the existing evaluator:

```bash
source ~/env/xca-cpu.sh
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF
cmake --build cbuild --parallel 4 --target \
  dense_small_n_convolution dense_small_n_evaluator \
  dense_small_n_convmat dense_small_n_recipe_reuse
```

Run serially, without other builds or timings. The recorded workstation run
pinned CPU 2; choose an available CPU on another machine.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bench=cbuild/benchmarks/dense_cpu_batched/experiments/small_n
taskset -c 2 "$bench/dense_small_n_convolution" > convolution_stages.csv
taskset -c 2 "$bench/dense_small_n_evaluator" evaluator_stages.csv > small_n_evaluator.csv
taskset -c 2 "$bench/dense_small_n_convmat" > convmat.csv
"$bench/dense_small_n_recipe_reuse" > recipe_reuse.csv
```

`dense_small_n_convolution --check-only` and `dense_small_n_convmat --check`
perform numerical checks without timing sweeps. The evaluator accepts an optional
case substring after the profile CSV filename, for example `r103/self_energy`.

## What each probe measures

- **Convolution:** production and an operation-for-operation clone, fixed 4x4
  products, real DLR GEMMs, and real DLR triangular solves. Policies change these
  independently. Arrays stay complex; real matrices act on real/imaginary parts
  as separate RHS columns, with no repacking or explicit interpolation inverse.
  Rank 13/103, batch 1/8/32, noncommuting smooth three-pole inputs. Five rotated
  timing trials target 5 ms per variant. Stage instrumentation runs separately
  from the uninstrumented medians. The helper also checks analytic convolution,
  partial-capacity workspaces, output aliases, and strided fixed-size products.
- **Evaluator:** the exact existing N=4 physical model and stratified sampling,
  self-energy/pairs/correlator, orders 2/5, ranks 13/103. Production scalar and
  production batch 1/8 are compared with separately named experimental
  evaluators. Five rotated trials each time one complete prepared call. All
  outputs are checked against the scalar result before and after timing.
  `fixed_all` changes convolution products, vertex products, endpoint products
  and correlator joins, not just vertices. `real_all` means real DLR GEMMs and
  real DLR solves; physical matrices remain complex.
- **Cached convolution matrix:** one fixed left operand shared by every batch
  lane, with eight changing sets of right operands. Compare production,
  `real_all_fixed4`, and a precomputed cppdlr `convmat`, including the packing
  required by batch layout. Seven rotated trials target 10 ms. Construction
  includes coefficient conversion and is measured separately, cold and warm.
  Alias rows include identical source restoration; speedup and amortization use
  a matching alias/noalias baseline. This is a kernel experiment, not a cached
  evaluator. No lookup, cache eviction, grouping or right-edge traversal is timed.
- **Recipe census:** untimed exact ordered edge recipes and completed initial
  prefixes in the existing physical fixture. Compare stratified samples with
  equally sized first/middle/last consecutive windows. Report global reuse and
  reuse within eight-backbone batches. Counts assume no numerical pruning and
  do not estimate cache lookup cost or weighted solver frequencies.

## Keep the experiment separate from production

`make_probe.py` generates a renamed copy of the current batched evaluator in the
build directory, with checked substitutions for timing scopes and arithmetic
hooks. It does not edit or replace production symbols. Assertions deliberately
fail if future source changes invalidate the experiment. The driver includes
the existing benchmark fixture so it cannot silently drift to a different model.
The generated evaluator adds no data ownership or production API decisions.

Stage timing in `evaluator_stages.csv` is a separate three-call instrumented pass.
`integrate_edge` includes `convolution`: subtract the latter to estimate edge
preparation, and never add both. Remaining unclassified time includes correlator
reflection/join, summation, allocation and orchestration. Compare percentages
within the instrumented pass; use the separate timing CSV for speedups.

Fixed-size kernels assume finite scientific inputs and a nonoverlapping output.
They use double precision, explicit real/imaginary arithmetic, and no fast-math.
They are N=4 prototypes, not a generally tuned small-matrix library. Other
dimensions, boundary-layer poles and solver-level workloads need validation
before adopting a production dispatch policy.

The detailed interpretation and measurements are in the
[workstation analysis](../../thinkstationP620_3945WX/small_n_analysis.md).

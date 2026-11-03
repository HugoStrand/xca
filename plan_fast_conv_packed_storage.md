# Fast convolution and packed evaluator storage

## Status and sequencing

Recorded 2026-11-03 on `dev_backbone`. First complete the structural cleanup in
`plan_block_sparse_refactor.md`. This document preserves the measurements and
implementation steps for the subsequent explicitly requested persistent goal.
Do not mix sparse API/helper extraction and packed-storage arithmetic changes
in the same code commit. Keep code, benchmarks, and work-log/plan commits separate.

The standalone complex, time-ordered `cppdlr::fast_convolve` and its six focused
tests are committed as `59c7bbd` (`[cppdlr] reuse convolution storage`). Dense
integration proceeds independently now, with helper and workspace owned by the
dense evaluator. CPU batched convolution and `dense::BatchedDiagramEvaluator`
are a parallel experiment, preserving the scalar evaluator for comparison.
Move immutable convolution tables to Action when sparse integration is ready;
mutable scratch remains evaluator-owned.

## Evidence and its limits

The comparison uses the **fixed** cppdlr routine (`b9b24ab`), which already
hoists the two full Hilbert transforms out of the DLR-node loop. Do not confuse
the additional gains below with the much larger improvement over that old bug.

Sources and reproducible commands:

- `benchmarks/fast_convolve/`: scalar convolution, including the benchmark-only
  reproduction of the old algorithm.
- `benchmarks/block_sparse_layout/`: prepared sparse numerical segment replays.
- `benchmarks/block_sparse_layout/thinkstationP620_3945WX/README.md` and CSVs:
  complete tables, numerical validation, timing spread, and hardware details.

Single-threaded Threadripper PRO 3945WX, Clang 19 Release, DLR rank 103:

| Forward segment dimensions | Current padded / allocating, us | Packed / output GEMM / fixed cppdlr, us | Packed / output GEMM / fast convolution, us | Combined speedup |
| --- | ---: | ---: | ---: | ---: |
| 1:1:1 | 265.30 | 256.92 | 171.12 | 1.55x |
| 2:1:2 | 306.51 | 296.87 | 194.33 | 1.58x |
| 6:4:6 | 1019.22 | 1003.68 | 795.44 | 1.28x |
| 4:16:64:16:4 (synthetic) | 39831.87 | 38954.53 | 25465.03 | 1.56x |

The physical test-model fixtures cover maximum sector dimensions one, two and
six, with both traversal directions. Synthetic cases cover growth/shrinkage,
rectangular intermediates, unused maximum capacity, and dimensions up to 64.
They are selected paths with deterministic dressed atomic propagators, not a
solver-weighted distribution of contributions. All 17 fixtures, both phases
(with/without convolution) and five variants passed; maximum relative error was
4.7e-14. The threshold was 2e-10 with explicit finite-element checks.

Conclusions:

1. Output GEMM into reused scratch makes physical vertex-only work 1.38--1.57x
   faster, but full numerical segments only about 1.01--1.03x faster.
2. Packing adds essentially no measurable full-segment gain on those physical
   fixtures, and approximately 2% on mixed-small synthetic cases. Its main
   purpose is a simple layout for direct DLR transforms and fast convolution.
3. The combined approach gives 1.28--1.58x on physical segments and about
   1.25--1.56x on synthetic mixed/reaches-64 cases. Fast convolution supplies
   most of the improvement.
4. Restricting self-energy scaling to active elements independently gives
   1.17x on the dedicated synthetic full-buffer-scaling control. Correlator
   segments already scale active elements, so this is not a universal gain.

Timings omit planning, operator lookup, edge-kernel multiplication, endpoint
sums, reflection/join, reset and final accumulation. They establish the
implementation direction, not complete evaluator or solver speedups. Small
percentages vary with workstation load and frequency. Future GPU designs target
high-end cards such as GH200; these CPU data and the local RTX A4000 do not
establish performance on those targets. No GPU experiments are part of this plan.

## Numerical and ownership design

For DLR rank `r`, a forward edge multiplies/convolves `(r,d,d)` with `(r,d,c)`;
a backward edge combines `(r,a,d)` with `(r,d,d)`. The anchor dimension stays
fixed over a segment while the path dimension changes. Self-energy endpoint
sums can change the anchor; correlator evaluation must retain one completed
segment while working on the other.

Current top-left slices of `(r,Nmax,Nmax)` have strides `(Nmax*Nmax,Nmax,1)`.
Packed views should instead describe `(r,rows,columns)` at the beginning of an
owned maximum-capacity buffer. Changing a view's shape does not preserve a live
packed input when output dimensions change: write vertex GEMM output to the
existing spare edge buffer, then swap buffer ownership and recreate views. The
old edge contents are dead at that point; fill the next edge afterward. Never
run an overlapping in-place GEMM, resize live owners, or reuse a stale view
across a swap.

Retain explicit active dimensions from BackbonePath. Keep the full forward
result live while the backward correlator uses its own result buffer and the
spare edge buffer. Preserve endpoint inputs before clearing/reusing output;
audit `Tkaps` and `Tmu` as well as `T`, `U`, and `GKt`. A saved endpoint slot
needs its own capacity-strided base address, with a packed active view inside
that slot; taking a top-left multidimensional slice is not sufficient.

The fast helper owns immutable complex DLR transform matrices and borrows the
mesh's unchanged interpolation object. The mesh currently shares that object
through a stable owning pointer. Verify copy/move lifetime at the Action
boundary before moving the helper there. Avoid copying atom_diag or adding a
second model-data representation.

Convolution scratch consists of five reusable buffers for coefficient arrays,
Hilbert-transformed coefficients and their product. Adapt the existing simple
workspace interface to make active packed views from capacity, rather than
caching a workspace for every `(m,n,p)` shape. A conservative sparse capacity
is `r*Nmax*Nmax` per buffer; dense is the fixed-dimension special case. Keep
rank, positive dimensions, BLAS-integer limits, capacity and aliasing checks.
The output may overwrite either input after both coefficient conversions;
workspace must remain disjoint from input/output.

At rank 103 and `Nmax=64`, the five buffers retain 32.2 MiB. This is not a peak
memory comparison with cppdlr's transient allocations. For much larger blocks,
record memory cost before allocating; maximum-capacity storage can waste memory
when most paths are tiny. Do not add a shape-cache architecture speculatively.
Models whose direct operator constructor supplies unknown dimensions need the
existing propagator/layout validation before deriving capacity.

## Implementation checkpoints after sparse refactoring

### 1. Reuse outputs and restrict operations to active data

Suggested code commit: `[bs] reuse vertex outputs`.

Replace allocating vertex `matmul` calls with output GEMM into spare storage,
and scale only live elements. Keep padded views and fixed cppdlr temporarily.
Preserve ordering, endpoint sums and reflection semantics. Validate before
changing the layout so allocation/API changes are independently reviewable.

### 2. Pack active sparse intermediates

Suggested code commit: `[bs] pack active workspace`.

Use contiguous active views into reusable capacity for both directions,
endpoint intermediates and correlator completion. Keep fixed convolution to
separate layout validation from the new convolution. No new public evaluator
API, hierarchy, shape cache or path-planning protocol is required.

### 3. Support varying active convolution shapes

Suggested code commit: `[cppdlr] reuse convolution capacity`.

Allow the existing scalar fast workspace to expose active rectangular views
whose sizes fit the reserved capacity. Preserve the fixed-size dense call
pattern and the independently developed batched overload. Test consecutive
shape changes, both legal output aliases, undersized capacity, layout/rank
errors and boundary-layer inputs. Keep the experiment-local CapacityConvolve
as a historical measurement implementation; production must use the tested
cppdlr extension rather than duplicate its numerical formula in evaluators.

### 4. Share immutable tables and adopt fast sparse convolution

Suggested code commit: `[eval] share fast convolution tables`.

Construct the immutable helper with Action's mesh after its lifetime is fixed.
Update dense, batched-dense if present, and sparse evaluators to use that helper
with their own mutable workspaces. Replace sparse convolution sites with the
output API. Do not share mutable workspace across evaluator instances or ranks.
Do not change Python constructors or perform hidden MPI collectives.

### 5. Validate full evaluations and measure integration

Use existing dense/sparse analytic, dynamical-interaction, prepared-path,
correlator, native-layout and MPI tests. Add focused cases for dimension growth
and shrinkage with retained endpoint data and preservation of the opposite
correlator segment. Retain returned results across later calls. Check Action/
evaluator copies keep the interpolation object alive. Do not relax numerical
tolerances merely to pass a structural or workspace change.

Rebuild bindings when headers change and run the full CPU suite. Keep the
existing benchmark results unchanged. Run comparable prepared-segment and
complete self-energy/correlator benchmarks, including orders with more than
one internal convolution. Record setup, reset, active dimensions, batch size
where relevant, and retained workspace bytes separately. Report numerical
agreement and end-to-end scope explicitly. Avoid a full large-system sweep;
bounded representative topologies and prepared contributions are sufficient.

## Restart checklist

Read this plan, `plan_block_sparse_refactor.md`, the corresponding work logs,
and current git status. Confirm the sparse refactor's validation checkpoint and
the fixed cppdlr dependency before beginning. Preserve user formatting and
unrelated files. Use the CPU environment switcher and existing build tree;
regenerate Python bindings only when required. Commit coherent code changes
and their explanations separately, with terse lowercase subjects. No pushing,
history rewriting or GPU allocation is authorized by this work item.

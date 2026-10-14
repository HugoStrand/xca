# Mapped kappa weighting

Implementation `ad4393b`. The mapper forms each mu buffer directly from owned
sparse column maps, avoiding materialized kappa matrices. It preserves field order,
uses the original left sparse application, and retains reference execution for
unsupported operators and singleton sets. Production code/defaults are unchanged.

All 28 isolated processes / 140 samples passed screening and exact replay.
Discovery counts match the reference and both bounded cache variants. Fusion can
change rounding relative to separate library primitives; errors remain within the
existing numerical gates, without a universal bitwise-equivalence claim.

Before timing, the fifteen-test native evaluator suite and two host classifier
cases passed (two CTest entries, 7.49 s). Seven focused memcheck cases passed with
zero errors and zero leaked bytes. They include complex rectangular primitives,
changed inputs/weights, empty columns, cancellation-sensitive order, a nine-point
four-field NCA cutoff neighborhood, metadata admission and unsupported operators
on accepted live paths. The two classifier cases also pass without CUDA. After
timing, native-disabled library-only CUDA independently passes the host classifier
and omits compiler-dependent mapper targets. This verification did not overlap timing.

[Manifest](manifest.json) pins sources, headers, binaries and commands.
[Build metadata](build_metadata.json), collected after timing, records the unchanged
local CMake configuration and CUDA compiler/architecture. [Samples](samples.csv)
aggregate the [raw GPU files](../../../block_sparse_cuda/thinkstationP620_RTXA4000/kappa-mapped-01/README.md).
[Summary](summary.csv) reports medians, errors and conservative storage bounds.
Each case has five timed repetitions in one process after warm-up; these are not
five independent process runs. Atom/model setup and original CPU control remain
separate. CPU control time is not a matched CPU throughput measurement.

| Modes / partition / order / indices | GPU reference ms | Two slots ms | Sixteen slots ms | Mapper ms | Mapper reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 / number / 1 / 0..2 | 16.904 | 14.455 | 11.747 | 4.519 | 73.3% |
| 4 / number / 2 / 0..1 | 10.805 | 9.852 | 8.779 | 6.191 | 42.7% |
| 4 / mixed / 2 / 0..1 | 14.032 | 13.570 | 13.442 | 11.470 | 18.3% |
| 4 / occupation / 2 / 0..1 | 14.961 | 15.146 | 15.046 | 15.194 | -1.6% |
| 6 / number / 2 / 0..1 | 30.544 | 27.586 | 21.766 | 12.673 | 58.5% |
| 8 / number / 1 / 0..2 | 138.598 | 124.232 | 87.055 | 30.395 | 78.1% |
| 10 / number / 1 / 0..1 | 351.544 | 341.258 | 294.326 | 187.962 | 46.5% |

Number-symmetry selections improve 43–78% over the ordered GPU reference; mixed
improves 18%. The singleton control regresses 1.6%, within the provisional protected
screen. These comparisons are local selected complete calls, not full topology-sum,
Solver, matched CPU or GH200 speedups. Maximum mapper screening relative error is
8.567e-15. All earlier baselines and experiments remain available.

[Analytic map allowances](analytic_maps.csv), without atom_diag construction or
N=14 evaluation, are 64,160 bytes for N10/r23 and 1,158,976 bytes (1.105 MiB) for
N14/r26. The prior full-set cache allowance at N14/r26 was 63.888 GiB. Maps remove
that extra requirement, while retaining the reference's 90.409 GiB variable
function/rank-buffer envelope. Fixed tables, sparse reserve and host/opaque allocator
storage remain additional; this is not a complete live peak or proof that N=14 fits.

Next qualify the same kernel with actual fourteen-mode Fock maps on bounded row
slabs, and refine large-sector memory scheduling. Higher-order cutoff-adjacent
cases, optional production/Python/MPI integration of this candidate, full Solver
throughput and target measurements remain pending. GH200-class hardware remains
the optimization target; no target/default dispatch change follows from A4000 data.
See [methodology](../../../../experiments/block_sparse_cuda/mapped_special/README.md).

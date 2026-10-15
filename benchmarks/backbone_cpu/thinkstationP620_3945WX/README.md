# ThinkStation P620 CPU measurements

Measured 2026-10-15 with the benchmark in code commit `9d515b8`, against the
production implementation at `dffa2b2`. CPU: AMD Ryzen Threadripper PRO 3945WX
(12 cores/24 threads); clang 19, Release, TRIQS 4.0.0. Numerical-library thread
counts were both one. No CPU affinity was pinned and frequency boost was enabled.
The workstation's GPU was not used. All fixtures have DLR rank 13.

The [benchmark description](../README.md) defines the phases and their overlap.
These are local cost diagnostics, not before/after speedups or full-solver timings.

## Times

Medians in **microseconds per sampled assignment**. Self-energy starts from an
already prepared assignment and a native block propagator; the single-particle
call includes public conversion/setup. Raw CSVs contain trial minima/maxima.

| Case | Flat preparation | Logical preparation | Signs + kernels | Paths + operand binding | Prepared self-energy | Single-particle range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [density4 / order 2](density4_order2.csv) | 0.209 | 0.150 | 0.528 | 3.653 | 594.673 | 459.151 |
| [density4 / order 3](density4_order3.csv) | 0.253 | 0.176 | 0.703 | 2.164 | 301.780 | 402.676 |
| [density4 / order 3 / longer run](density4_order3_long.csv) | 0.252 | 0.172 | 0.694 | 2.060 | 301.684 | 401.495 |
| [density4 / order 6](density4_order6.csv) | 0.328 | 0.224 | 1.239 | 1.494 | 43.855 | 153.654 |
| [n_partition_spin_xyz / order 2](n_partition_spin_xyz_order2.csv) | 0.206 | 0.150 | 0.475 | 2.389 | 802.972 | 595.923 |
| [n_partition_spin_xyz / order 3](n_partition_spin_xyz_order3.csv) | 0.245 | 0.172 | 0.657 | 2.509 | 1217.420 | 874.657 |
| [unequal_sets_nA0 / order 2](unequal_sets_nA0_order2.csv) | 0.217 | 0.162 | 0.505 | 2.003 | 346.828 | 246.753 |
| [unequal_sets_nA0 / order 3](unequal_sets_nA0_order3.csv) | 0.249 | 0.172 | 0.720 | 1.710 | 371.674 | 306.049 |

The measured recipe preparation is 0.15–0.33 microseconds. On these samples it is
below 1% of a complete prepared self-energy call. Path discovery plus binding is
also a small fraction at orders two and three. This supports keeping the clear
preparation boundary before pursuing allocation reuse or caching. It does not
identify the remaining numerical bottleneck: the complete calls also include DLR
transforms/convolutions, intermediate allocations, result packing, and resets.
A phase profiler would be needed to split those costs.

Prepared-versus-flat complete-call differences are comparable to noise and are
not evidence of a measurable speedup. At order six most sampled assignments
have no path, so the lower complete-call time reflects pruning, not cheaper
high-order nonzero diagrams.

## Coverage and memory

| Case | Sectors / max dimension | Total assignments | Sampled assignments / candidate paths | Range calls per kind | RSS before → after (KiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| density4 / 2 | 16 / 1 | 32 | 32 / 384 | 2,560 | 21244 → 21312 |
| density4 / 3 | 16 / 1 | 512 | 64 / 200 | 5,120 | 21320 → 21392 |
| density4 / 3 / longer | 16 / 1 | 512 | 64 / 200 | 20,480 | 21188 → 21260 |
| density4 / 6 | 16 / 1 | 2,097,152 | 64 / 10 | 5,120 | 21104 → 21176 |
| n_partition_spin_xyz / 2 | 5 / 6 | 56 | 56 / 312 | 4,480 | 21940 → 22012 |
| n_partition_spin_xyz / 3 | 5 / 6 | 1,568 | 64 / 293 | 5,120 | 21788 → 21864 |
| unequal_sets_nA0 / 2 | 6 / 2 | 32 | 32 / 192 | 2,560 | 21156 → 21220 |
| unequal_sets_nA0 / 3 | 6 / 2 | 512 | 64 / 217 | 5,120 | 21212 → 21284 |

RSS increased by 64–76 KiB after warm-up across these processes. The four-times
longer order-three run had the same 72 KiB increase as the shorter run. The
order-six run samples a space of 2,097,152 assignments without storing that space.
These observations are consistent with bounded streaming and allocator warm-up;
they do not establish a platform-independent memory ceiling or exclude every
possible leak. Correlator storage can still grow with the model's sectors and
block dimensions, and operator support grows with model size and pole count.

## Reproduce

After the build/environment setup in the parent README:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for model in density4 unequal_sets_nA0 n_partition_spin_xyz; do
  for order in 2 3; do
    cbuild/benchmarks/backbone_cpu/backbone_cpu "$model" "$order" 8
  done
done
cbuild/benchmarks/backbone_cpu/backbone_cpu density4 6 8
cbuild/benchmarks/backbone_cpu/backbone_cpu density4 3 32
```

Each command is a fresh process. Retain the CSV output when comparing later
experiments. Broader physical models, productive high-order cohorts, large
sectors, and GH200 execution require separate measurements.

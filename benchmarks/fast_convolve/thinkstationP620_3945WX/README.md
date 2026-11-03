# ThinkStation P620 / Threadripper PRO 3945WX

Measured on 2026-11-03 with an AMD Ryzen Threadripper PRO 3945WX (12 cores,
24 hardware threads, approximately 31 GiB RAM). The benchmark used a single
OpenBLAS thread, `OMP_NUM_THREADS=1`, Clang 19.1.1 and a Release build. The GPU
was not used.

The installed CPU stack contains TRIQS `2930120`, NDA `4e59077`, and cppdlr
`b9b24ab` (`prq_convolve_opt`). The cppdlr baseline already includes the fix
that moves the Hilbert transforms outside the per-node matrix-product loop.
The original cppdlr algorithm is reproduced by the benchmark-only
`legacy_convolve.hpp`, retaining the redundant full Hilbert transforms inside
the DLR-node loop. The installed library remains fixed. The fast implementation
is an uncommitted XCA draft. These measurements are for the standalone
convolution kernel, not the diagram evaluator.

The DLR basis has **rank 103**, cutoff `100000`, tolerance `1e-12`, and inverse
temperature seven. All three paths include conversion of both inputs from values
to coefficients, convolution, and output assignment. Basis, immutable helper
and reusable workspace construction are excluded. The sweep covers every
power of four from `4` through the requested upper bound of `512`; its largest
dimension is therefore `256`.

| Matrix dimension | Original algorithm | Fixed cppdlr | Fast draft | Fixed speedup over original | Fast speedup over original |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 7.8364 ms | 0.3198 ms | 0.2377 ms | 24.50× | 32.97× |
| 16 | 81.094 ms | 3.5113 ms | 3.0919 ms | 23.10× | 26.23× |
| 64 | 1361.863 ms | 68.996 ms | 61.294 ms | 19.74× | 22.22× |
| 256 | Skipped | 2201.595 ms | 1475.559 ms | — | — |

The fixed routine removes the repeated Hilbert transforms, changing that work
from `O(r^3 D^2)` to `O(r^2 D^2)`; the matrix products still cost
`O(r D^3)`. The fast draft provides a further 1.13–1.49× improvement over the
fixed routine in this sweep. These are total timings including coefficient
conversion, so the speedup need not equal the DLR rank.

For `D=256`, the original algorithm was skipped before its first call: quadratic
scaling from its `D=64` timing gave a screening estimate of **21.79 seconds per
call**, above the three-second guard. This is an estimate, not a measurement;
it also omits the extra cubic growth of the matrix products.

| Matrix dimension | Original calls per batch × batches | Fixed calls per batch × batches | Fast calls per batch × batches | Fast relative maximum error |
| --- | ---: | ---: | ---: | ---: |
| 4 | 2 × 5 | 60 × 5 | 78 × 5 | 1.26e-15 |
| 16 | 1 × 5 | 5 × 5 | 6 × 5 | 1.01e-15 |
| 64 | 1 × 1 | 1 × 5 | 1 × 5 | 8.69e-16 |
| 256 | — | 1 × 1 | 1 × 1 | 1.21e-15 |

Five-batch results report the median per-call time after warm-up. Cases with
one batch exceeded the 500 ms threshold and report a single measured warm-up
call, with correspondingly greater timing uncertainty. Outputs were checked
separately before timing. The original algorithm had zero output discrepancy
against the fixed routine at every measured dimension; the fast draft's errors
relative to that same reference are shown above.

Basis construction took 163.4 ms and the fast helper's matrix preparation
took 0.261 ms. The fast draft uses five reusable full-size complex work arrays;
these occupy about 32.2 MiB at `D=64` and 515 MiB at `D=256`. These results do not
establish the effect on an end-to-end evaluator or on other input spectra.
The optional `D=1024` case remains subject to the runtime and memory guards
described in the benchmark README and was not part of this sweep.

See [timings_with_legacy.csv](timings_with_legacy.csv) for this three-way run.
The earlier fixed-versus-fast run remains in [timings.csv](timings.csv).
The [benchmark README](../README.md) describes input construction, build
instructions, measurement boundaries and guards.

## Dense evaluator integration: numerical roundoff

A separate diagnostic compared the fixed cppdlr convolution with fast convolution
inside the complete dense evaluator. The dense evaluator sources from XCA commit
`59c7bbd` were compiled in `/tmp` under the class name `FixedDiagramEvaluator`,
alongside the current fast dense evaluator and the unchanged block-sparse
convolution. The installed cppdlr remained `b9b24ab`; no production source was
changed for this comparison. The model, maximally crossing topologies, and
large fixture ordinals come from `test/c++/flat_index_range.cpp`.

The diagnostic used beta two, cutoff 40, 20 fitted poles, and the same single-thread
CPU environment as above. At tolerance `1e-10`, the DLR rank is 23. The two order-six
contributions have maximum magnitudes `2.06349e-8` (self-energy) and `1.75431e-7`
(correlator).

| Comparison at order six | Self-energy relative maximum error | Correlator relative maximum error |
| --- | ---: | ---: |
| Fixed dense vs block sparse | 0 | 1.34e-31 |
| Fast dense vs fixed dense | 2.32e-13 | 1.61e-13 |

Across orders two through six at the same mesh tolerance, the largest fast/fixed
discrepancy was `7.47e-13`. At order six with mesh tolerance `1e-12` (rank 26),
the discrepancies were `5.61e-13` and `3.49e-13`. These differences arise after
changing the LU solve and transform arithmetic; the former fixed dense/sparse
agreement benefited from sharing that arithmetic.

As an additional check, each result was interpolated to nine common relative
times `0.1, 0.2, ..., 0.9` and compared with the fixed result using mesh tolerance
`1e-14`. The original `1e-10` mesh differed by `1.24e-8` for self-energy and
`3.51e-7` for the correlator, relatively. Switching to fast convolution changed
these comparisons by roughly `1e-13`, well below the mesh-refinement differences.
The finer-mesh result is a convergence reference, not an exact analytic answer.
At tighter mesh tolerances `1e-13` and `1e-14`, fast/fixed differences reached
`4.78e-12`, so no universal relative-error bound is inferred from this fixture.

The large-index test therefore uses **`2e-12` relative tolerance only for
cross-evaluator comparisons on its fixed rank-23 mesh**. Its integer counts,
index-bound checks, and same-evaluator vector/pair/correlator comparisons retain
their previous requirements, including `1e-13` for numerical API identities.
Analytic correctness tests were not relaxed.

[dense_integration_roundoff.csv](dense_integration_roundoff.csv) records the
complete diagnostic. Its first table contains per-mesh algorithm comparisons;
its second table contains the common-time-grid refinement comparisons.

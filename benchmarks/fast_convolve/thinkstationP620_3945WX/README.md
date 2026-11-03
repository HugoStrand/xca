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

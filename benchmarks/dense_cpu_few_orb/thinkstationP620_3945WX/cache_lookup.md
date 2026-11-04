# Canonical edge keys and hash lookup

Measured on 2026-11-04 on the Threadripper PRO 3945WX, using Clang 19.1.1,
Release `-O3 -DNDEBUG -std=c++23`, system OpenBLAS and one thread pinned to CPU 2.
No native-architecture or fast-math flags. Both `OMP_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` were set to 1; no builds or tests ran concurrently.
The measured production change is commit `cf28480c`; controls use the same
numerical evaluator with the cache policies described below.

The [comparison driver](../cache_lookup.md) isolates three cache policies with
the same FewOrb numerical evaluator: the previous ordered linear lookup,
canonicalized linear lookup, and canonicalized `std::unordered_map` lookup.
Canonicalization sorts `(pole_index, argument_sign)` pairs without discarding
repeated factors. It also uses this order when forming the numerical operand.

## Result

Keep canonicalization and the single hash lookup path. Complete order-two
calls show no material regression: the new timings range from 0.04% slower to
0.71% faster than ordered linear lookup. A separate tiny-cache policy is not
justified by these measurements. Disabled-cache and single-entry checks also
show no material overhead, including order-five products with multiple factors.

The main benefit is fewer distinct numerical operands. Two-pole order-five
samples need 15 entries instead of 31. The larger ten-pole sample needs 487
instead of 1,161, reducing payload from 7.37 MiB to 3.09 MiB. With insufficient
capacity, canonicalization improves hit rates as well: at rank 103 and an
8 MiB budget, both policies retain 159 entries, but misses fall from 683 to 336
per 512-backbone call. Overall evaluation improves by about 3.6–3.8% there.

Hashing adds a modest 0.1–0.6% over canonical linear lookup in the ten-pole
measurements; most of the gain here comes from canonicalization. It avoids a
linear scan as the key set grows without adding a second production policy.
These results do not imply a large end-to-end acceleration from lookup alone.

## Timings

Median **milliseconds per call** over seven rotated trials. Order two is a
complete topology evaluation including backbone preparation. Order five uses
bounded samples from the fully crossing topology, with prepared backbones;
it is a cache stress case, not a complete high-order topology sum.

| Rank | Poles | Order / backbones | Observable | Budget | Ordered linear | Canonical linear | Canonical hash |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 13 | 2 | 2 / 16 | Self-energy | 8 MiB | 0.734 | 0.731 | 0.734 |
| 13 | 2 | 2 / 16 | Correlator | 8 MiB | 0.418 | 0.416 | 0.418 |
| 103 | 2 | 2 / 16 | Self-energy | 8 MiB | 7.269 | 7.248 | 7.271 |
| 103 | 2 | 2 / 16 | Correlator | 8 MiB | 5.217 | 5.191 | 5.180 |
| 13 | 10 | 5 / 512 | Self-energy | 8 MiB | 50.123 | 49.571 | 49.410 |
| 13 | 10 | 5 / 512 | Correlator | 8 MiB | 40.102 | 39.508 | 39.333 |
| 103 | 10 | 5 / 512 | Self-energy | 8 MiB | 612.741 | 592.715 | 590.918 |
| 103 | 10 | 5 / 512 | Correlator | 8 MiB | 545.728 | 527.244 | 524.745 |
| 103 | 10 | 5 / 512 | Self-energy | 64 MiB | 591.973 | 585.633 | 584.381 |
| 103 | 10 | 5 / 512 | Correlator | 64 MiB | 528.526 | 521.345 | 520.755 |
| 13 | 10 | 5 / 4096 | Self-energy | 8 MiB | 400.408 | 390.934 | 389.468 |
| 13 | 10 | 5 / 4096 | Correlator | 8 MiB | 320.017 | 311.342 | 310.101 |

The 512-backbone ten-pole samples encounter 314 ordered keys and 222 canonical
keys when capacity permits. These counts describe the sampled workloads, not
all possible keys. In particular, the larger cache does not guarantee every
possible operand of the full topology is retained.

All 66 result rows across 22 cases pass comparison with the baseline dense
evaluator; the largest relative maximum discrepancy is **9.42e-14**, against a
2e-10 limit. Sorting changes floating-point multiplication order, so equivalence
is numerical rather than bitwise. Both canonical variants have identical cache
counts and numerical errors. Payload accounting and reported speedups were
checked independently of the driver.

Timings include rebuilding numerical cache contents on every public call,
canonicalization, lookup, hash-node/key allocation, and result allocation.
Previously allocated numerical buffers and hash buckets are retained. Initial
numerical-storage allocation and Action construction are outside timing.
Generated control classes compile separately from the production class, so
sub-percent differences should not be overinterpreted as portable improvements.

## Reproduction

Run the default driver for [the 512-sample sweep](cache_lookup.csv). For
[the two larger samples](cache_lookup_large.csv), run it separately with:

```text
r13/p10/prepared/self_energy/order5/cache8388608 4096
r13/p10/prepared/correlator/order5/cache8388608 4096
```

The raw CSVs include cache hits, misses, entries, bytes, timing ranges and errors.
The older result files in this folder used ordered cache keys and remain
unchanged. The current self-energy API sums both special-arc directions.

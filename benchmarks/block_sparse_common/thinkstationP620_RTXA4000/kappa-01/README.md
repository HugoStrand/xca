# Bounded kappa product retention

Timing implementation `f3a66d8`. This E6 experiment compares the retained ordered
CUDA reference with private two-slot and sixteen-slot kappa caches. Products are
retained only within one special vertex; weighted mu/kappa summation order is
unchanged. The sixteen-slot cap covers all fields in these bounded number-symmetry
fixtures; it is not an unlimited cache. Singleton sets allocate no cached products.

All 21 isolated processes / 105 samples passed numerical screening and exact replay.
Discovery counts, warm output norms and original-CPU control errors match the
reference in every corresponding row. The initial ten-test suites for both variants,
four focused memcheck cases (two per variant), and CPU-only configuration checks
completed before timing. Memcheck reported zero errors/leaked bytes. An additional
complex off-diagonal bosonic-set case was qualified after timing, separately from
measurements: final eleven-test suites pass in both ordinary CUDA and native-disabled
library-only CUDA builds; the bosonic memcheck case also has zero errors/leaks.
Native checks run only when that backend is available. Production code is unchanged.

[Manifest](manifest.json) pins commands, source/header/binary hashes and machine.
[Samples](samples.csv) aggregate the raw [GPU files](../../../block_sparse_cuda/thinkstationP620_RTXA4000/kappa-01/README.md).
[Summary](summary.csv) contains medians, errors and conservative storage bounds.
Each case has five timed samples in one process after a warm-up; these are not five
independent process repetitions. Setup, original CPU control and full synchronized
API call times remain separate. CPU control timing is not a matched CPU benchmark;
this table compares GPU routes only.

| Modes / partition / order / indices | Reference ms | Two slots ms | Sixteen slots ms | Sixteen-slot reduction |
| --- | ---: | ---: | ---: | ---: |
| 4 / number / 1 / 0..2 | 16.836 | 14.288 | 11.506 | 31.7% |
| 4 / number / 2 / 0..1 | 10.955 | 9.921 | 8.854 | 19.2% |
| 4 / mixed / 2 / 0..1 | 14.243 | 13.470 | 13.558 | 4.8% |
| 4 / occupation / 2 / 0..1 | 15.040 | 15.192 | 15.135 | -0.6% |
| 6 / number / 2 / 0..1 | 30.828 | 27.853 | 21.770 | 29.4% |
| 8 / number / 1 / 0..2 | 136.789 | 125.910 | 86.314 | 36.9% |
| 10 / number / 1 / 0..1 | 352.720 | 340.414 | 296.941 | 15.8% |

NCA two-index cases include both line directions. Number-symmetry full retention
reduces selected call latency by about 16–37%; partial retention gives smaller
savings. Mixed cases have modest gains. The singleton occupation control regresses
about 1%, within the provisional protected-case screen. This is local evidence
for an optional memory-bounded route, not a production/default or GH200 decision.

The [analytic memory table](analytic_memory.csv) uses binomial block dimensions,
without atom_diag construction or an N=14 evaluation. It records the reference's
variable envelope (two functions plus thirteen maximum rank buffers) and additional
cache allowance. It excludes fixed tables, sparse reserve, host/opaque allocations.
Ranks are 23 for the benchmark mesh and 26 for the retained fourteen-mode inventory
proxy. Extra cache allocation uses actual rectangular extents at runtime; these
worst-case envelopes deliberately use the largest square rank buffer.

| Modes / rank | Reference variable GiB | Two-slot additional GiB | Full-set additional GiB |
| --- | ---: | ---: | ---: |
| 10 / 23 | 0.410 | 0.044 | 0.218 |
| 14 / 26 | 90.409 | 9.127 | 63.888 |

Cache capacity increases rejected-workload risk. In particular, full retention is
not an appropriate universal large-sector policy; the fourteen-mode reference
already needs further streaming/admission work. Next investigate direct mapping
for partial-permutation kappa operators and fused ordered weighting, avoiding
materialized q products where semantics permit, while keeping general CSR fallback
and the current controls. Non-permutation operators, correlators, target hardware
and complete Solver/topology throughput remain unqualified optimization regimes.
GH200-class hardware remains the target, not this RTX A4000.

See [generic method and build instructions](../../../../experiments/block_sparse_cuda/special_vertex/README.md).

# Edge-cache lookup comparison

This controlled evaluator benchmark compares the original ordered linear cache,
a canonicalized linear cache, and the production canonicalized hash cache. The
first two evaluators are generated in the build tree from the current FewOrb
source, replacing only cache fields and lookup/preparation. There is no extra
production configuration API. The generation script asserts its source hooks so
an incompatible evaluator change fails instead of silently invalidating the
comparison.

```bash
source ~/env/xca-cpu.sh
cmake -S . -B cbuild -DUpdate_Python_Bindings=OFF
cmake --build cbuild --target dense_few_orb_cache_lookup --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  cbuild/benchmarks/dense_cpu_few_orb/dense_few_orb_cache_lookup > cache_lookup.csv
```

An optional first argument filters the case name, and a second changes the
prepared-backbone sample count (default 512). For example, `r103/p10` selects
the higher-rank, ten-pole cases. Run without MPI, pinned to one physical CPU
where practical, with no competing builds or benchmarks.

All cases use the shared N=4 model at DLR ranks 13 and 103. Complete order-two
topologies measure the low-order cost including backbone preparation. Order-five
cases use the fully crossing topology and a stratified deterministic sample of
prepared backbones, bounded independently of the full diagram count. Both
self-energy pairs and fermionic correlators are covered. The original two-pole
input is supplemented by a ten-pole input: alternating signs, magnitudes
`0.35 + 0.13*p`, and repeated original coefficient slices scaled by `2/10`.
The Hamiltonian, propagator and operator lists remain the shared fixture's.

Cache budgets are 8 MiB throughout and 64 MiB for the ten-pole cases. Separate
low-order cases cover zero capacity and one retained operand; order-five
self-energy cases also cover zero capacity to measure sorting overhead without
operand reuse. Numerical payload
budgets are identical among variants. Key metadata is excluded, as in the
production API. CSV hit/miss/entry counts show whether timing differences arise
from key deduplication, lookup cost, or cache pressure. Canonicalization sorts
integer `(pole_index, argument_sign)` pairs and retains repeated factors. The
sorted factors are also used to build the numerical operand, so rounding changes
can occur even in the cache-disabled case.

Every variant is compared with the baseline dense evaluator before and after
timing, requiring finite nonzero output and relative maximum error below
`2e-10`. Seven trials rotate variant order; calibration targets 10 ms with one
to eight repetitions. Timings include lookup, key canonicalization, filling a
fresh numerical cache, output allocation and all evaluator arithmetic. Model,
Action and prepared-backbone construction are outside timing; backbone decoding
is included for complete topologies. Allocations are warmed, but valid numerical
operands never survive a public-call boundary. Results belong in hardware-named
subdirectories.

These are bounded evaluator measurements, not complete high-order topology sums
or solver iterations. Canonicalization's value depends on overlap and pole
diversity; hashing's value depends on the number of retained entries. Compare
the two canonical variants to isolate lookup choice.

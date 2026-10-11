# E4 results: N4 products and pointwise fusion

Custom complex-double4×4 products pass the original analytic/rotation suites and
strict CUDA-reference benchmark gates. Combined with128-lane E3 batching they give
14.0–14.8× CPU throughput on the primary r16/order6–8 cases. Production is unchanged.
The experiment implements ordinary SE only; unsupported methods in test adapters
still use the production evaluator. Promotion needs a separate review.

| N4/r16 crossing |Three-process warm API medians |CPU speedup |
|---|---:|---:|
|Order6,1 index, capacity1 |2.377 /2.376 /2.377ms |Still slower than CPU |
|Order8,1 index, capacity1 |3.443 /3.355 /3.357ms |Still slower than CPU |
|Order6,128 indices |6.818 /6.869 /6.815ms |14.00–14.23× |
|Order8,128 indices |9.147 /9.197 /9.135ms |14.46–14.80× |

Same dense-small-v2 seed/index lists and precision. CPU ratios use matching rows in
each process; frozen CUDA599.061/831.037ms gives roughly87–91× at full128 width.
Against E3 batch alone32.7/44.5ms, custom products add about4.8×. Per-index API time
is53–54us/71–72us. Primary runs record samples, MAD, hashes, libraries and commands.
Timing source revisiona30708d; later regressions add dense complex basis/changed-G
coverage and a contiguous-weight guard without changing accepted kernel arithmetic.

Controls separate the effects: mode0 retains E3 library operations; mode1 fuses
special-vertex weighting, separate parity multiplication and accumulation, plus final
convolution addition/scaling; mode2 adds N4 products. B1 order6 mode0/1 costs
3.018/2.964ms; order8 4.207/4.150ms. Pointwise savings alone are under2%, and B128
mode1 order8 is44.406ms, essentially E3. No material pointwise-only win is claimed.
Capacity1 gives an independent custom-product control; capacity128 is E3+E4.

Warm capacity128 padding remains expensive: one index costs5.65/7.96ms,8 indices
5.89/8.24ms,32 indices6.06/8.43ms,128 indices6.81/9.14ms (orders6/8).129-index tails
cost17.70/23.93ms because two padded chunks repeatedly miss the single-chunk schedule.
Use a capacity close to the actual chunk size. Dynamic tail widths/multiple schedule
ownership were not implemented; the discontinuity is measured, not concealed.
Uncached128 calls cost10.80/14.30ms, retaining metadata preparation in API time.

Fresh first calls cost10.75–10.96/14.39–14.43ms; setup4.48ms in repeat01. Reuse saves
about4.0/5.2ms per call. These are initialized-process costs: a separately recorded
CUDA context probe is156.45ms, and validation warms common libraries. Adding that
probe to setup/first cost gives about2 full-batch calls to amortize against matching
CPU. Standalone cold-library initialization is not fully measured. B1 remains slower
than CPU even after amortization; graphs do not remove its serial numerical work.

Seventeen runs/250 backend-case groups pass strict CUDA parity; sampled errors are
zero, including individual lanes, aggregates, changed G, duplicates and empty lists.
Coverage reaches order12, ranks16/26/36, second seed31415 and tails. Rank26/36 controls
are one process each, not repeated speedup claims. Their full128 order6/8 medians are
12.62/17.67ms and18.81/25.94ms; rank36/order12 is40.35ms. Known high-order CPU-relative
discrepancies remain in CSV (rank36 max3.95e-6); low-order CPU agreement passes.

Six CTest entries pass under mode2, including all four API regressions, independent
wide DLR primitives and custom product/pointwise checks. A dense complex Hilbert-basis
rotation and same-address changed propagator are checked through order12 at capacities
1/8/128. CUDA memcheck reports0 errors for API and custom primitive tests. Production
CPU-only support is unchanged; all23 protected source hashes still match064b64e.

Rounding controls matter. Disabling FMA or reversing weighting operands passed the
analytic tests but failed high-order parity. Input-first complex multiply with normal
CUDA FMA matches DGMM exactly. An interleaved complex custom dot failed high-order
checks; grouped real dot products with explicit __fma_rn pass. No precision/tolerance
was relaxed. Negative and positive controls are retained under results/validation/.
N4 kernels use128 threads,40 registers/thread, no shared scratch or register spills;
pointwise kernels use24/16 registers. No occupancy/peak-FLOPs claim is made.

Nsight order6 B128:396 kernels,5.85–6.22ms active,6.62–7.18ms traversal (profiling
adds variability). The44 custom tiny products/vertices take about0.40ms versus E3's
26.37ms. Forty retained rank GEMMs take1.16ms; solves3.57–3.94ms now dominate.
B1:243 kernels,2.17ms active,2.44ms traversal, solves1.25ms. Both traces have zero
inner allocation/free/explicit host transfers. B128 has6295552D2D and8192UVM bytes;
B1 has57344–61440D2D and no measured UVM copies. Known arrays221248B/23629888B;
observed peaks51310261B/74747573B include references/library caches, not residency.
Full measured API retains two malloc/free calls, two benchmark device fences,
two stream fences and two upload-event handoffs. Runtime/geometry CSVs are retained;
raw traces stay in ignored cbuild-small-profile/fused/.

The trace-justified custom pivoted-solve follow-up is isolated in ../fused_solve/.
It reconstructs accurately but fails strict public parity, so retain library LU.

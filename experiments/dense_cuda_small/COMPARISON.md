# Small-matrix CUDA exploration: comparison and recommendation

For ordinary self-energy on N4/r16, combine cross-index batching with the checked
custom4×4 products (`fused_small`, mode2).128 indexed evaluations take6.8/9.2ms at
orders6/8, versus matching CPU96–97/133–135ms:14.0–14.8× across three fresh processes.
The frozen CUDA implementation takes599/831ms. Single evaluations still favor CPU.
These are experimental opt-in implementations; production remains at064b64e.

| Strategy |Order6/8 single-index API |Order6/8 at128 indices |Decision |
|---|---:|---:|---|
|Frozen CUDA |4.4 /6.2ms |599 /831ms |Retained reference |
|E1 owned stream |3.05–3.39 /4.24–4.34ms |363 /511ms |Useful foundation; CPU faster |
|E2 graph replay |2.82–2.85 /3.91–4.01ms |338 /464ms |Modest incremental benefit; fixed-key reuse |
|E3 shared-LU batching |3.01 /4.20ms at capacity1 |32.7 /44.5ms |About2.9× CPU at full width |
|E4 pointwise fusion |2.96 /4.15ms at capacity1 |~E3 |Under2% alone; no material win |
|E3+E4 custom N4 |2.38 /3.36–3.44ms at capacity1 |6.8 /9.2ms |Best checked throughput,14.0–14.8× CPU |
|Custom triangular solve |Strict parity fails |Primitive faster; API rejected |Keep library LU |

Each row's [strategy report](README.md) identifies matching cases, source revisions,
first/setup costs, repeatability, memory and traces. E1/E2 throughputs are historical
measurements; their upload-ordering fix was revalidated with three fresh quick runs.
Do not infer every cross-row difference from a single noisy process. E3+E4 uses
capacity1 versus capacity128 controls to distinguish independent product improvements
from cross-index parallelism; it is a measured combination, not an automatic dispatch.

For experiments, use `fused_small` with XCA_FUSED_MODE=2 and XCA_BATCH_CAPACITY close
to the number of submitted indices, up to128. Submit many indexed evaluations together
when they share topology/model data. Full batches amortize the same serial backbone
work across independent lanes; shared LU/rank transforms avoid redundant factorizations.
Avoid padding128 lanes for one index.129 indices at capacity128 costs17.7/23.9ms,
versus6.8/9.2ms for128: a padded second chunk and cache misses are real costs.
Changing-index preparation is charged; uncached128 calls still cost10.8/14.3ms.

Fixed-key graphs help submission overhead but do not increase arithmetic density;
their creation amortizes after several repeated calls. Pointwise fusion alone is small.
The original batched tiny GEMM kernel does too much work for4×4 blocks: E3 trace
26.37ms falls to about0.40ms with custom products. Preserving grouped real FMA order
was essential. After that, library solves dominate. The custom-solve probe reconstructs
at~1e-16 but fails higher-order CUDA parity, despite being faster in isolation. Small
residuals alone do not establish acceptable diagram accuracy; keep the stricter gate.

Constructor/first-call costs are included separately from warm API samples. CUDA
context probes (~156–162ms) are separate, and validation warms common libraries:
the benchmark is not a full standalone cold-start measurement. Adding the recorded
context probe estimates about2 full128 calls to amortize E3+E4 against CPU. First
call plus constructor wins in an already initialized process. No peak GPU occupancy,
all-dimension speedup, or universal CPU win is claimed.

Correctness: reused C++ analytic/complex-rotation tests, strict per-lane and aggregate
CUDA comparisons through order12, ranks16/26/36, tails, duplicates, empty inputs,
long indices and same-address input changes pass for the recommended experiment.
Dense complex basis regressions and sanitizer checks pass. Ordinary SE only; pairs,
SPGF, dynamic interactions/correlators remain production paths. Known high-order
CPU differences are reported, and no tolerance or precision was relaxed. A pageable
upload race found by wide primitives was fixed with explicit event handoff in all
owned-stream experiments. All23 protected production files remain unchanged.

Next useful work after review: adaptive chunk/tail widths and a small bounded cache
of schedules; ordered device accumulation to reduce128 lane gathers/axpys; reusable
host-facing staging to remove two complete-API allocations; graph+fusion as a launch
control for small batches. Solve optimization requires matching blocked arithmetic or
separate numerical-stability work before acceptance. Optional CPU hoisting and other
lower-priority routes were not required for this exploration and remain unimplemented.
The current experiments are retained; promotion needs its own review and full supported
method tests. No production change or push was performed.

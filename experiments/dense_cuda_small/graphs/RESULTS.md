Upload-ordering note: this report describes the original experiment revision. E3
exposed a pending pageable-upload race at larger sizes; event-based handoff has
been added. Earlier benchmark inputs passed parity but are insufficient evidence
for the general stream contract. Post-handoff tests/timings supersede these where
available; preserve the original data as historical evidence.

# E2 graph replay results

CUDA graph capture is feasible with E1's owned adapters and gives a modest further
warm-call improvement. It does not address the4×4 throughput gap. Production and
E1 implementations remain intact; graph use is opt-in and bounded to128 indices.

| N4/r16 crossing B1 | Graph medians, three processes | Frozen CUDA speedup |
|---|---:|---:|
| order6 |2.859 /3.132 /2.833ms |1.41–1.56× |
| order8 |3.949 /3.923 /4.017ms |1.55–1.59× |

Order8 improves another6.5–8.5% over the three E1 repetitions. Order6 has more
between-process spread and one graph repetition is slower than its E1 counterpart;
do not describe every case as a graph win. All exact comparisons and raw/MAD data
are retained. Neither strategy approaches CPU's .7–1.0ms single-index latency.

Graph B128 warm medians are338.2ms at order6 and464.0ms at order8, versus E1
362.9/511.0ms and CPU92.44/127.97ms. First graph calls cost529.3/739.3ms: capture
scales with the whole indexed schedule. Against E1 the measured extra first-call
plus constructor costs are155.7/235.5ms, recovered at warm savings24.73/47.00ms
per call after about8/7 total calls. Changing index/topology contents rebuilds the
single cached graph, so repeated reuse is needed for these incremental benefits.
First-call timing measures a cache miss on a fresh evaluator; rebuilding an existing
cache additionally destroys its old graph, and is not timed separately here.

B1 order6, first repetition: setup3.468ms, first call3.899ms, warm2.859ms. E1
setup3.439ms, first3.105ms, warm3.091ms. This case needs about5 total calls to
recover the graph's additional first/setup cost against E1. Costs vary by case;
complete raw first/setup samples are available, without correctness prewarming.

Seven benchmark runs cover210 backend/case groups: three quick processes,
throughput B1/8/32/128, bounded order12 coverage, rank26 and second seed31415.
All strict CUDA-reference checks pass with exact sampled output parity, including
per-index/aggregate, duplicate/empty, and changed G at the same address. Low-order
CPU checks pass; high-order tiny-term CPU discrepancies remain separately reported.
No tolerance was loosened. Four isolated CTest entries pass: existing one-/two-fermion
and complex-rotation tests with an indexed graph adapter, plus API cache/lifetime
regressions. Memcheck on all three API regressions reports zero errors.

Node-level N4/order6 replay trace retains257 kernels and53248bytes of device copies
per index. Traversal has no allocation/free or explicit host/device copies, one
graph launch and one completion stream fence. Full API still has two malloc/free
calls and the benchmark's two device fences. Profiled node-level traversal wall is
3.076–3.132ms and active kernels2.762–2.765ms; profiling overhead is not an
unprofiled bound. Managed status migration remains131072bytes per call. Known owned
array/pointer/status bytes370496 exclude opaque graph/library allocations; observed
trace peak51,459,509bytes includes reference and library caches, not physical
candidate residency. Graph/key ownership is bounded to one cached schedule.

The initial Nsight graph-level trace hid replay kernels and copies: its zero-kernel
CSV is retained as explicitly incomplete evidence. Profiling was repeated with
`--cuda-graph-trace=node`; the auditor now rejects any traversal with no visible
kernel activity. Raw reports/SQLite remain ignored under cbuild-small-profile/graphs/.
First graph capture/instantiation is outside the replay traversal audit and remains
inside the charged complete API range (`xca.graph.build`).

Conclusion: graph replay helps fixed-schedule reuse, with a setup penalty for changing
indices. Retain it as a possible combination with batching, not as a substitute for
E3. No automatic production promotion; proceed to wide shared-LU cross-index batching.

Upload-ordering note: this report describes the original experiment revision. E3
exposed a pending pageable-upload race at larger sizes; event-based handoff has
been added. Earlier benchmark inputs passed parity but are insufficient evidence
for the general stream contract. Post-handoff tests/timings supersede these where
available; preserve the original data as historical evidence.

# E1 results: reduce synchronization with an owned stream

Ordinary SE is correct and meaningfully faster than the frozen CUDA baseline, but
still roughly4× slower than CPU at N4. This experiment reduces waiting/submission
cost; it does not batch independent indices or change kernel arithmetic.

| N4/r16 crossing, B1 | Frozen CUDA median | E1 medians, three processes | Speedup range |
|---|---:|---:|---:|
| order6 |4.407ms |3.091 /3.065 /3.437ms |1.28–1.44× |
| order8 |6.236ms |4.241 /4.255 /4.277ms |1.46–1.47× |

Cases match dense-small-v2 seed1729 exactly. Full raw samples, first-call/setup
costs, medians/MAD, library/binary/source hashes and comparisons are in `results/`.
The order6 third process was slower; report the range instead of choosing its best
sample. No universal CPU speedup is claimed.

Throughput remains nearly linear: order6 B128 costs362.9ms (2.835ms/index), versus
CPU92.44ms (.722ms/index); order8 B128 costs511.0ms (3.992ms/index), versus CPU128.0ms.
This is a useful graph/batching prerequisite, not the hoped-for small-matrix result.

Seven final runs cover210 backend/case groups: three quick repetitions, throughput
B1/8/32/128, bounded orders through12, rank26 control, and second seed31415. All
experimental per-index/aggregate/changed-input/duplicate/empty comparisons pass
strict CUDA-reference parity; sampled outputs are bitwise identical. Low-order CPU
checks pass. Known tiny high-order CPU-relative discrepancies remain baseline
limitations and are reported in CSV; no tolerance was loosened.

Reused analytic one-/two-fermion and complex-rotation suites pass for experimental
ordinary SE. The test adapter delegates unsupported methods to production CUDA.
Reused DLR primitive tests pass, plus invalid-prefix recovery, >INT_MAX indices and
independence from NDA's shared solver-synchronization setting. Five CTest entries
pass; CUDA memcheck on the API regressions reports zero errors.

N4/order6 measured-call trace:257 kernels, one owned-stream fence in traversal,
zero traversal allocation/free events and zero explicit host/device copies.
Complete API has two device synchronizations (benchmark fences), one stream fence,
and two malloc/free calls, retained from baseline host-facing materialization.
Four complete measured calls include the first plus two samples and one unreported
warmup. Kernel active time2.806–2.813ms; profiled traversal wall3.181–3.190ms.
Asynchronous phase NVTX ranges measure host submission and must not be interpreted
as GPU phase time. The full traversal encloses actual completion.

Managed solver-status metadata migrates131072bytes per call (maximum61440bytes per
copy); this is visible, not dismissed as zero-copy. E1 owns370496bytes of arrays and
pointer/status metadata. Trace-observed allocated peak51,459,509bytes includes the
reference evaluator, library caches/handles and context-visible allocations; it is
not physical GPU residency or a precise standalone candidate peak. Raw Nsight
report/SQLite stay under ignored cbuild-small-profile/stream/; audited CSVs are
tracked. Owned cuSOLVER handle overhead and status migration remain optimization
opportunities.

Conclusion: retain E1 independently. Its arithmetic-preserving enqueue path is a
candidate prerequisite for graphs and index batching. Prioritize batching for the
remaining throughput gap. Production evaluator is untouched and E1 is not promoted.

Historical timing-source hashes match revision74f2adf. The only later build-recipe
change added the separate API regression target; its measured recipe is retained as
results/measured_CMakeLists.txt. It did not change the timed executable sources.

Example setup amortization (order6 B1, first repetition): constructor3.439ms versus
baseline2.093ms; first call3.105ms versus4.413ms. About two repeated calls recover
the1.346ms extra setup at this case’s1.316ms warm-call saving. This is a case-specific
example; all first/setup samples remain available rather than hidden in warm medians.

## Upload-handoff revalidation at3d7c826

Three fresh quick processes pass strict CUDA parity after the event handoff.
Order6 B1:3.048/3.391/3.062ms; order8:4.240/4.339/4.342ms.
Raw samples, source/binary hashes and matching baseline comparisons are in
results/serial_stream-handoff-quick-{01,02,03}/. The relevant analytic/API/primitive
CTest suites also pass (summaries retained under index_batch/results/validation).
These results validate the repaired stream contract; earlier measurements remain
recorded at their original revisions.

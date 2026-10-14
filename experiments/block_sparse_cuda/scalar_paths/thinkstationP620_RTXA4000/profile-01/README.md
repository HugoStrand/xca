# Scoped scalar allocation trace

GH200 remains the performance target. CUDA tracing captured only the second
(warmed) scalar SE range call. Setup, CPU control and first GPU warm-up are outside
the range. cudaProfilerStop/export heavily affects the wrapper-reported call time;
**none of these times is an unprofiled benchmark**. CUDA API counts and relative
categories diagnose mechanisms; API intervals include profiler overhead.

| API | Fan-in calls | Contiguous arena calls |
| --- | ---: | ---: |
| cudaMalloc | 3206 | 137 |
| cudaFree | 3208 | 139 |
| cudaMemcpy | 1121 | 99 |
| cudaMemcpy2D | 2055 | 8 |
| cudaStreamSynchronize | 1253 | 230 |
| cudaMemsetAsync | 1029 | 6 |

GPU arithmetic in the first trace occupied about 3.7 ms, whereas allocation/free
and small copy API intervals dominated runtime categories. Arena changes single-call
input/output storage and transfers, retaining the same scalar recipe and ordered
accumulation. The count reduction confirms the targeted mechanism; it is not proof
of a target speedup or complete peak admission.

Raw textual reports are [fanin_stats.csv](fanin_stats.csv) and
[arena_stats.csv](arena_stats.csv). [manifest.json](manifest.json) pins profiler
version/options, exact case, executable/report hashes. Large binary reports remain
at the recorded temporary paths and may disappear; no binary report is committed.
The baseline trace predates the arena source change; binary hashes identify the
actual captured executables, rather than pretending both used one code revision.

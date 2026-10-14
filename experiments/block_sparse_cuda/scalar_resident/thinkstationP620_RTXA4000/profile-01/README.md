# Scoped resident weight trace

Warmed N6 OCA, first1/count8, explicit multi-set fixture. Setup and CPU screening
excluded; profiler wrapper/export times are not benchmarks. Capture uses the same
CUDA-only profiler-API range as the earlier multi-set trace. All numerical screens
and repeatability passed. Nsight2026.3.2; binary/report hashes retained.

Host-to-device events remain115, but payload falls from7,883,316 to1,014,324 bytes
(7.77x reduction). Device-to-host588,848 bytes/16 events and device-to-device9,067,520
bytes/49 events are unchanged. Runtime span12.098ms, summed API intervals8.101ms,
kernel intervals2.486ms. Old trace span25.890ms includes pre-hoisting parity work,
so that span change cannot be attributed solely to residency. Interval sums overlap
and tracing affects durations. Independent unprofiled residency speedups are in
the common density-resident-01 campaign.

Alloc/free156/158, streamSync272 and eventRecord/streamWait258/258 counts remain.
Residency reduces payload and host weight preparation, not API frequency. Next
use owned reusable descriptor/staging buffers and fewer synchronization points;
host-source lifetimes and stream ordering must remain qualified. No GH200 inference
or promotion follows. Binary reports remain at recorded /tmp paths; compact text
stats and provenance are retained. No target allocation was added.

# Scoped upload trace

Warmed N6 occupation/density OCA, first1/count8, rank23. Nsight2026.3.2 CUDA-only
profiler-API range, sample/cpuctxsw disabled. CPU numerical screen and exact replay
pass. Wrapper time includes export and is excluded from performance claims.

Resident→staging allocations156→69/frees158→71, default-stream eventRecord258→165;
staging adds initial descriptor copies and retains operation completion. Staging→
events streamSync278→185, eventSync0→87, eventRecord165→261. Kernel count unchanged;
HtoD1,051,188bytes/118, DtoH588,848/16 and DtoD9,067,520/49 identical for both new
variants. Extra source-completion events preserve host lifetime before rewrites.
All runtime/kernel interval sums and spans are tracing-affected, not speedups.

Source revisions and report hash retained; binary reports in /tmp may disappear.
Timing evidence is in block_sparse_common local density-streams-01. GH200 remains
the target; these traces do not qualify target performance or production policy.

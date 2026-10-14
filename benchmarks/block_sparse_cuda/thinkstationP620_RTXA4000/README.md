# ThinkStation P620 / RTX A4000 CUDA reference

`reference-01/` contains raw synchronous CUDA reference calls, with one warm-up
and bounded explicit index windows. All 15 processes completed and passed numerical
screening against CPU. Paired timings, provenance and limitations are recorded in
[block_sparse_common](../../block_sparse_common/thinkstationP620_RTXA4000/README.md).
GH200-class accelerators remain the target; these results do not set dispatch defaults.

[Matched CPU/CUDA calls](../../block_sparse_common/thinkstationP620_RTXA4000/hoisted-cpu-02/README.md)
use raw rows in `hoisted-cpu-02/`. `hoisted-cpu-01/` timings are excluded because
they overlapped compilation; numerical checks are retained.

# GH200 bounded phase qualification

Source edb8b73, recipe290d65f, pinned GTest988ea2c. Job3847180 on n412 COMPLETED0:0
in138 s, one GPU, fifteen-minute cap, compile concurrency capped at six. GCC14.3 /
CUDA12.9.86, observed GH200 FB memory97871 MiB (95.58 GiB). Reused installed Arm TRIQS;
source-only isolated staging, timestamps normalized before transfer, hashes checked,
no shared-checkout update/install. Raw job/build/tests/accounting under
[qualification records](../../../gh200_qualification/Arrhenius_GH200/capacity-01/README.md).

Both CTests pass12.80 s: fourteen active stream evaluator GoogleTests (including all
four library/ordered and reference/mapped selections with dynamic interactions, plus
arena incompatibility) and the small independent phase control. No skips. No broad
suite/Python/MPI qualification follows. Reduced129x128x127/rank64 phase and actual
N14 central/adjacent sector-shaped3432x3003x3432/rank64 phase both pass sampled oracle
and exact sampled replay. The latter uses a simulated opposite fragment and32 MiB
operator reserve, not a physical atom/model or complete Solver.

Large explicit payload90493714692 bytes (84.28 GiB); sampled global free-memory delta
90503839744 bytes. Initial setup/upload21.031 s, independent oracle4.022 s, first/warm
convolution1.756/1.691 s. Per-process wall28.72 s and MaxRSS467136 KiB (456.19 MiB).
These times distinguish phase work from setup/oracle and include no unmeasured Solver
throughput claim. A complete host propagator/output/model would add other host storage.

Three spatial entries at each of64 rank nodes compare against independent long-double
sums using full inner/rank dimensions. Max error1.290e-22 at sampled norm3.629e-10,
about3.55e-13 relative, below the predeclared1e-11 relative +1e-24 screen. Other entries
are not independently checked. Replay samples are exact and final scratch release leaves
only tables. No cutoff/physical-rank accuracy/full-atom/full-observable-capacity proof.
Global delta is sampled, not a continuous/process-local peak. This establishes a bounded
large phase on actual target hardware; complete evaluator composition/admission and
Python/MPI integration remain gates before promotion.

Allocation138 GPU-seconds (.03833 GPU-hours), tracked cumulative1704 s (.47333 nominal
hours) including prior cancelled attempt. Shared monthly100 GPU-hour ceiling also covers
other work; latest preflight projinfo3.22 hours/30days is explicitly unreliable. Do not
interpret tracked total as account balance. No duplicate/remaining job or new install.

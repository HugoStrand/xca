# Local composed NCA qualification

Source268f255. Native CTest passes0.59 s, library-only0.57 s; CPU configure
omits the CUDA-only target. Five execution cases pass: N1, N4 one/all fields,
N10 one/all fields. All use two calls, complete output-entry checks and exact
repeated diagonals. N10 uses Lambda4000/rank64 without truncation. One field
is exactly equal to the CPU/projector oracle; ten fields differ by at most
2.222e-16 at oracle norm1.21714. Ten-field first/warm calls4.265/4.224 s;
one-field0.702/0.655 s. These are fixture timings, not physical Solver speedups.

Host/device budget rejections return2, N14 workstation guard and malformed
suffix return1. N14 dry-run reports38.26 GiB/function,78.66 GiB device and
99.48 GiB host bound. It does not execute the large case locally. Small all-field
memcheck reports zero errors/leaks. The initial absent CSR source-column metadata
failure is retained; corrected before source commit. During development the CLI
Lambda conversion to mesh omega_max was corrected before final qualification.
Formatting-only rebuild/retest followed the full local screens and memcheck.

GH200-class is target. Target evidence belongs in its own hardware folder;
local screens do not establish target execution, physical DLR accuracy, generic
mixed-matrix behavior, or higher-order memory. Previous broad suites and MPI remain
separate; this change adds an isolated experimental probe, not evaluator arithmetic.

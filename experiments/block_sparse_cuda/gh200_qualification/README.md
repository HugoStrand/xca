# GH200 qualification

Arrhenius allocation: 100 GPU-hours/month. Use focused single-GPU jobs and retain
build caches/results before considering sweeps. Record allocated GPUs and walltime;
a queued or scheduler-validated job is not numerical qualification. Do not repeat
failed builds without diagnosing their logs. Avoid full matrix-size/order/topology
Cartesian products and full fourteen-mode atom_diag initialization as first probes.

`smoke.sbatch` requests one GPU for at most ten minutes (nominal ceiling 1/6 GPU-hour),
with no whole-node/extra CPU/memory request. It compiles an isolated pinned source
snapshot using the installed Arm CUDA TRIQS stack and runs four focused block-sparse
device/DLR/observable/admission CTests. Python/bindings and benchmarks are excluded
from this first build. It neither installs software nor changes shared checkouts.
Source dependencies must be supplied as pinned source-only archives under `src/deps`.

Submit from a dedicated project qualification directory containing `src`, `logs`
and the job script. First run `bash -n` and `sbatch --test-only`; then record the
returned job ID for an authorized submission. Inspect state/logs before retrying.
Compiler concurrency follows allocated CPUs, capped at six. Actual GPU allocation
and elapsed time must be recorded from Slurm accounting, separately from the cap.

After smoke qualification, decide on a small representative benchmark cohort from
local evidence, reuse the Arm build, and predeclare a bounded additional budget.
No broad GH200 speedup or full large-system qualification follows from smoke tests.

The checked-in script's absolute Slurm log paths refer to the isolated
`shared/qualification/xca-block-sparse-3f462c3` directory. Update those literal paths
when preparing another snapshot; do not overwrite an existing snapshot's sources.
The source revision and dependency hashes belong in `source-revisions.json` beside
the submitted script. Git source archives omit `.git`, so CMake's empty Git hash
is expected; the recorded archive revision is authoritative.

`cohort.sbatch` reuses that snapshot/build, enables benchmarks, and builds only the
CPU, library CUDA and ordered CUDA frontends. It caps allocation at fifteen minutes
on one GPU. `run_cohort.py` records seven explicitly bounded workloads and three
samples per backend, with per-process timeouts and immediate failure stop. The cases
cover scalar rejected/active windows, retarded mixed correlators and number-sector
SPGF/SE through N=10. Each CUDA frontend checks its own rotated CPU control and exact
repeatability. Backend processes run sequentially with single-thread BLAS/OpenMP.
Runner and binary hashes are recorded alongside pinned source revisions. This is
an initial cohort, not a statistically established dispatch crossover or a full
topology sum. Raw CPU/GPU and combined results belong in the respective benchmark
folders under `Arrhenius_GH200/initial-01`.

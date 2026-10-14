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

`scalar.sbatch` reuses the same pinned tree, builds the constants/arena/fan-in
candidate and its matched CPU control, and first runs all 14 scalar semantic tests.
It then invokes `run_cohort.py --cohort scalar`: six cases, four backends, three
samples. The unchanged CPU and ordered GPU remain controls. Work includes scalar
zero/active windows, an eight-index range, N=10 scalar blocks, an order-six chain
window and a mixed retarded correlator fallback. Inspect actual survival before
claiming high-order arithmetic coverage. Allocation is capped at fifteen minutes
on one GPU; failures stop the campaign. Results use `Arrhenius_GH200/scalar-01`.

`density.sbatch` targets isolated snapshot 6987c4d, preserving the earlier snapshot
and Arm build. It builds four matched density-set frontends and the new multi-set
semantic suite plus four production CUDA controls. Tests run before the bounded
four-case/48-sample density campaign; any failure stops benchmarking. One GPU is
capped at fifteen minutes (nominal maximum 0.25 GPU-hours), build workers capped
at six from allocated CPUs. Installed Arm TRIQS is reused; no software is installed.
Its literal Slurm paths refer to `shared/qualification/xca-block-sparse-6987c4d`.
Prepare pinned GTest source and source-revisions.json as for the first snapshot.
Remote syntax/scheduler checks, live queue/balance inspection and actual submission
are still required; local preparation does not establish target qualification.

`resident.sbatch` qualifies pinned snapshot 77f59ac in a separate directory. It reuses
the installed Arm TRIQS stack, builds the resident-weight candidate and matched
parity-hoisted CPU/GPU frontends, and runs nineteen candidate controls plus four
production CUDA CTests before the four-case/36-sample resident campaign. One GPU,
fifteen-minute cap (0.25 nominal GPU-hours); no install or shared-checkout update.
Record actual job status and allocated time before claiming target qualification.

`real_nca.sbatch` is prepared for isolated source snapshot b5a131c. It builds and
qualifies the production pointwise NCA policy and retained prototype before running
`run_cohort.py --cohort real_nca`: four cases, four routes, three samples (48 rows).
Routes are the constants CPU control, general ordered CUDA, retained real-NCA
prototype and production. Cases include N4/N8 real scalar NCA and higher-order/
number-sector fallback. CPU control preparation differs from the unchanged CPU
frontend; keep that distinction in comparisons. This is not a full topology sum,
large-system qualification, Python/MPI target test, or universal crossover table.

The script retains one GPU / fifteen minutes (at most .25 nominal GPU-hours),
uses the installed Arm TRIQS stack, and installs nothing. Literal log paths point
to `shared/qualification/xca-block-sparse-b5a131c`. Stage pinned b5a131c sources
and GTest as before, copy the updated runner/job recipe separately, and record all
archive/runner hashes. Local syntax/help checks and local cohort replay do not
prove remote scheduler acceptance or execution. Live SSH/account/queue checks and
`sbatch --test-only` remain required before exactly one authorized submission.

Before running `real_nca.sbatch`, normalize regular files in the locally staged
`src` to epoch 946684800 (2000-01-01 UTC), then write `source-mtime-epoch.txt` with
that value and transfer with timestamps preserved. The job validates this marker
instead of rewriting thousands of shared-filesystem timestamps in GPU time. A
partial build from the same pinned source/toolchain may be reused after its job is
terminal; retain the earlier record. This change addresses the observed filesystem
wait in the first NCA target attempt, not a numerical or compiler failure.

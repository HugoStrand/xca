# GH200 complete-sector NCA capacity

Pinned source268f255, corrected recipe50ef2a6, isolated Arm build and installed
TRIQS reused without prefix/shared-checkout changes. One GH200120GB on n444,
fifteen-minute allocation cap, no CPU/memory overrides. Source archive and recipe
hashes match. Reduced all-field N4 CTest passes1.54 s, N10/rank64 one-field
complete call passes0.157 s. No tolerance relaxation or rank truncation.

N14/rank64 one-field **complete production evaluator call** passes: all15 number
sectors returned, every matrix entry checked against the CPU two-state/projector
oracle, max error0, norm0.12171436906137988,28 contributing paths. Each complete
host function41079398400 bytes/38.26 GiB. Explicit device bound84463062276/78.66 GiB;
sampled device free-memory delta88613584896/82.53 GiB. This difference includes
opaque CUDA/library allocation costs excluded from the declared array bound.
Host estimate106818289664/99.48 GiB, measured process MaxRSS92312704 KiB/88.04 GiB;
granted host memory105048 MiB/102.59 GiB. Slurm batch MaxRSS691904K measures a
different process/sampling scope and is not used as the large probe peak.

Setup2.363 s, complete evaluation32.235 s, full-entry verification8.239 s,
large process wall43.07 s. One target sample; no target replay/performance crossover
claim. Local two-call controls remain separate. The fixture has diagonal inputs,
one active flavour, one pole and NCA only. It does not qualify physical N14
atom diagonalization, DLR accuracy, general off-diagonal inputs, all-field target
costs, higher-order composed capacity, or target Python/MPI.

First job3850983 exited2 after7 s before configure: the partition exports memory
per CPU and omitted SLURM_MEM_PER_NODE. Corrected admission reads actual AllocTRES;
second job3851037 COMPLETED0:0 in117 s. Both retained, including failure recipe.
Scheduler test-only IDs3850930/3851026 were not allocations. Combined124 GPU-seconds,
tracked cumulative1828 seconds/0.50778 nominal GPU-hours. Shared100 GPU-hour/month
ceiling applies to all user work. Latest projinfo3.26 hours/30days includes other
work and is explicitly marked unreliable by the site. Preserve this bounded result;
do not repeat it solely for more timing samples.

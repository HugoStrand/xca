# GH200 MPI phase qualification

Job3854536 completed0:0, parent/batch28seconds and overlapping extern29seconds,
one GPU/two ranks on n535, accountnaiss2026-3-569-gpu, five-minute cap. Scheduling
probe3854532 was test-only. All three process groups pass; step0/1/2 take10/5/5seconds.
No rebuild: binary-check.txt confirms qualified source7b5d223/runtime85ecb2d extensions
unchanged. Benchmark0a55b82 and recipe2713bb0 are staged separately from shared sources.
Exact environment, private interpreter PMIx profile, source hashes, recipe, raw launcher
logs and accounting retained. No installation, module-prefix or shared-checkout update.

Fixed number, mixed-retarded and occupation-tail windows compare ownership-only
canonical reductions with the Solver range/block-reduction pattern. Every worker
asserts size2 and passes scalar allreduce before evaluator construction. Actual site
command is srun --mpi pmix --ntasks=2; independent serial processes are not counted.
GPU cases run sequentially. Profiling is a separate local control, not part of this job.

[Results and limitations](../../../../../benchmarks/block_sparse_common/Arrhenius_GH200/mpi-01/README.md).
Parent elapsed brings tracked work to3384nominal GPU-seconds/.940000GPUh, within the
shared100GPUh/month ceiling. This is not a statement of remaining site quota or other
user usage. All jobs/steps terminal. Raw downloaded hashes are in the paired workstation
report's raw-sha256.json before archive whitespace normalization.

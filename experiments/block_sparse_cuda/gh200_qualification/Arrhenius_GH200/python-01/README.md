# GH200 Python facade and split MPI qualification

Evaluator snapshot7b5d223 (runtime85ecb2d), pinned c2py58e0586/adapold3c4ab7/
pyed248bd34/GTest988ea2c. Isolated Arm build against existing CUDA TRIQS;
GCC14.3, Python3.13.5, NumPy2.3.2. Three required extensions build; imported
module origins resolve to this build. Ordered/Mapped/RealNca capabilities true.
Native sparse ON, dense-small OFF. No install/shared source or prefix mutation.
The imports transcript includes a standalone-MPI warning before its JSON object.

Job3851249: Py_cuda_backends passes1.98s (14 methods,3 dense-small skips),
Py_block_sparse_cuda passes63.62s (12 active methods), two CTests65.63s.
MPI then fails before workers because mpprun is missing from the reset PATH.
Job3851450 uses absolute site helper but auto-detection chooses plain srun:
two independent size1 processes, rejected by the rank guard. Job3851572's
minimal native-mpirun profile obtains size2 but stalls without a completed test;
cancelled after249s. Diagnostic gdb attach did not obtain a usable stack before
cancellation; no causal stack inference is made. Original recipes/logs retained.

The final private interpreter alias reuses the installed OpenMPI .hpcinfo srun
profile: --mpi pmix, block placement, FI_PROVIDER=cxi, SLURM_MPI_TYPE=pmix.
The alias must resolve to the configured CTest interpreter. Site helper profile
selection is explicit; every worker verifies size2, scalar allreduce and barrier
before CUDA setup. Extension hashes match the successful serial build. No shared
Python executable or MPI configuration is edited. Runtime evaluator unchanged.

Job3851765: **partial run**, nine methods complete on both ranks (18 OK results),
then the420s command cap expires inside storage_policies_all_observables. No full
12-method target MPI pass is claimed. Job3851926 selects only the three unfinished
methods, all pass per rank in323.619s; launcher/Slurm exit0. Coverage.json records
exact method sets. Together the transcripts cover every block-sparse method on
target, including observables/dynamics, archives, policies, self-consistency,
rank-local errors and recovery. This is **split coverage**, not a completed
single-process-lifetime twelve-method run. Timed faulthandler dumps are diagnostics,
not failed assertions. No tolerance/input/shape reduction was made to fit time caps.
The one-GPU/two-rank stress workload has substantial elapsed cost; this is not a
performance result or a measured attribution to context switching.

All five one-GPU jobs terminal. Parent elapsed168+11+249+432+339=1199 seconds,
including failures/builds/timeouts, nominal0.33306 GPUh. Tracked total3027 seconds,
0.84083 nominal GPUh. Raw parent/step/cleanup accounting retained; site billing
may differ. Shared100 GPUh/month ceiling includes other user work. Earlier
3.26h/30day projinfo includes other work and is site-labeled unreliable.

No new C++ broad-suite/target all-target/full physicalN14/higher-order capacity
claim. Prior local broad suites, full two-rank local suite, and bounded target
N14 capacity are separate evidence. Preserve successes; do not repeat full costly
MPI suites solely to replace a partial transcript with a summary. Remaining work
is the requirement-by-requirement implementation/completion audit and any concrete
missing evidence it identifies. GH200 remains optimization target; A4000 a proxy.

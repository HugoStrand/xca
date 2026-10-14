# MPI qualification launcher controls

The initial runner passes two selected storage/collective-recovery methods on
both ranks in16.515/16.533 s. A requested communicator size3 with two real ranks
fails before tests, as required. The interpreter alias extension passes the
archive/metadata method on both ranks in0.357/0.360 s. The final allreduce/barrier
preflight plus verbose method passes on both ranks in0.359 s. Each worker verifies
its actual communicator size, not just Slurm/launcher metadata.

The recorded CTest environment resolves the build-tree package and cached pyed/
adapol sources. The alias must be the same file as the configured interpreter.
These are focused launcher controls on RTX A4000, not a repeated full MPI suite
or GH200 evidence. Target launch/profile results belong in the GH200 folder.
No production evaluator arithmetic changed. Raw launch commands and original text
hashes are retained with archive-only whitespace normalization.

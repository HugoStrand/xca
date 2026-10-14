# Scalar upload completion events

Private experiment targeting GH200-class hardware, retaining scalar_staging and all
older candidates. Local RTX A4000 evidence screens the design only.

Reuse the staging buffers but replace completion after each descriptor, constant or
pointer kernel with an upload-completion event. Before rewriting a host buffer, wait
for the event recorded immediately after its preceding copy. Device reads and the
next device-buffer write remain ordered on the same evaluator stream. Three events
are created lazily per cohort and reused. This permits host preparation while the
preceding device kernel may still run; pageable uploads may still block internally.

Complete the recipe before NDA host copy-back. Existing copy/zero/BLAS/DLR methods,
initial default-stream upload ordering, fan-in completion and staging destruction
retain their barriers. This is a scoped experiment, not an asynchronous evaluator
API. Exceptions drain the owned stream before source/device storage and events are
released. Memory admission is unchanged from staging; event resources are separate
runtime bookkeeping, not additional array payload.

`block_sparse_scalar_streams_tests` inherits nineteen controls and adds repeated
source-buffer updates plus an exception while readers may be queued. The stress
control verifies each retained target and gather result after destructor completion.
`run_cohort.py --cohort streams` compares matched CPU, staging GPU and this GPU on
four bounded density cases. `profile_density_streams` captures one warmed call;
profiler durations must not be used as benchmark timings.

Promotion still requires numerical/cutoff policy, target evidence and production/
Python/MPI integration gates. No old candidate is replaced by this experiment.

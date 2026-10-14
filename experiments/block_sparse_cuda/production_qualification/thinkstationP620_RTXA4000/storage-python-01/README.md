# Storage policy facade qualification

Local RTX A4000 evidence; the optimization target remains GH200-class hardware.
Source: 85ecb2d, including stream-ordering fix 7054f0c. Policies are explicit;
Resident remains the default. No dispatch or performance conclusion follows.

Final native checks: 22 C++ storage controls (50.18 s), Py_cuda_backends
(1.59 s), and 12 Py_block_sparse_cuda methods (67.77 s); three CTests pass
in 119.54 s. Native-disabled library checks pass in 52.47 s (0.92/51.54 s).
CPU metadata/archive checks pass in 0.69 s: two active methods, ten CUDA skips.
Two local MPI ranks each pass all 12 methods in 331.449/331.454 s; launcher exits 0.
Constructor-bound probes cover all four static/dynamic real/complex signatures
and all three storage policies. Archives, old defaults, mutable preferences,
observables, collective failures and recovery are exercised.

Retained failures include the C++ exception mapping expectation, an MPI launcher
using the installed package instead of CTest ENVIRONMENT_MODIFICATION, and a real
HostInputs numerical race. NDA default-stream complex initialization could overwrite
owned-stream input copies. The production fix orders those streams before copying;
focused exact comparisons and complete MPI tests subsequently pass. No tolerance
was relaxed. Regenerated dense-small bindings retain optional-build guards.

This is focused local qualification, not a new full-suite or GH200 Python/MPI run.
No Arrhenius allocation was made for these tests; tracked cumulative target usage
remains 1704 GPU-seconds (0.47333 nominal GPU-hours). Original text hashes are
retained alongside whitespace-normalized archive hashes in manifest.json.

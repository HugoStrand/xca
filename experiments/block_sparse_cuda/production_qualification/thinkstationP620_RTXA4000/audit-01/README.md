# Audit verification gaps

Source2f3191c adds tests, with no evaluator arithmetic/API change. GPU target remains
GH200-class; these focused checks run on RTX A4000. No cluster allocation used.

A deliberate PTX device trap is submitted on DeviceOps' owned stream in a fresh
re-exec child. Launch must succeed; synchronization must throw, and the sticky
runtime status must be launch failure or illegal instruction. All three previously
working storage-policy evaluators must then throw, returning no CPU result. The
child exits without unwinding poisoned GPU owners; process teardown releases its
context. Parent GPU zero/synchronize succeeds afterward. Native and library-only
fault tests pass0.98/1.05s. This tests real asynchronous fault propagation and
subsequent poisoned-context rejection; it does not inject a fault mid-topology or
claim recovery of a poisoned CUDA context. The intentional trap is labeled
cuda_fault_injection and excluded from sanitizer-clean cohorts by design.

TRIQS preserves fundamental-operator insertion order, so the existing shuffled
consecutive-label test already exercised mode reordering. It now additionally uses
nonconsecutive/negative integer labels across three block names, retaining
independent bit-map/CAR and dense comparisons. This is strengthened label coverage,
not evidence that the earlier reordering test was invalid.
Full native sparse-operator entry passes six controls0.56s; CPU host entry passes
0.45s, with CUDA fault case disabled (zero active GPU tests on CPU).

Sector-state reversal transforms every field, dynamic operator, endpoint and
propagator consistently. NCA/OCA SE, SPGF and bosonic correlators must obey basis
covariance at the existing1e-13 tolerance. Library, ordered, HostInputs and
StreamFragments variants pass four GoogleTests in three native CTests/3.47s.
This exercises the exported snapshot's arbitrary state ordering; atom_diag's
read-only Fock metadata is not modified. Library-only three focused CTests pass
2.23s: fault, permutation and noncanonical labels. Native final fault/host entries
pass1.54s. All GPU runs sequential. Permutation memcheck6.005s reports zero errors,
zero leaks; the deliberate fault is not passed through memcheck.

No new full-suite, MPI, benchmark-performance or target-hardware claim. Existing
broad suites/target split MPI remain separate evidence. The audit still identifies
missing MPI scheduling/reduction phase timing and fresh-process repetition evidence;
three timing samples in one process do not satisfy that latter plan gate.

# Production C++ storage-policy qualification: RTX A4000

Code9524c0c, frozen reference98a5ebd. Resident remains default. HostInputs borrows host
blocks and uses completed phase scratch; StreamFragments adds two-slot recomputation.
All return complete observables. Explicit RealBatch requires Resident. Planner and
runtime bounds agree across output policies; optional strided input packing is CPU
storage, at most one maximum-rank block, outside device bounds. New observer reports
successful fragment builds. Shared helpers preserve arithmetic across choices.

Final native policy CTest22 GoogleTests (one host-only inventory) passes50.79 s;
frozen stream experiment14 controls passes13.80 s, two CTests64.59 s. Earlier default
reference/memory/new-policy/Python CUDA checks pass five CTests81.75 s. Python checks
validate resident entry points after C++ constructor change, not unexposed policies.
Library-only passes three CTests43.72 s with20 active storage controls and one native-
policy skip, then added cutoff control9.053 s. Both native flags OFF, Build_Benchmarks
restored ON after independently configuring/building the CUDA test without benchmarks.
CPU two CTests pass0.03 s: host inventory works, CUDA-only case disabled (no GPU proof).
All configured CPU/native CUDA targets build, including retained experiments. Library
build is focused targets, not a newly claimed all-target/full-suite pipeline.

Two ordinary lifetime/admission controls pass memcheck232.746 s, zero errors/leaks;
strided-host control separately passes5.502 s, zero errors/leaks. NaN-filled unused
columns catch accidental contiguous-byte copying. Exact comparisons cover frozen
reference, changed G/masks, complete outputs, exact budgets/rejection/recovery,
all sparse/special selections with dynamic interactions, and reused original evaluator
suites at higher orders/correlators/tiny roots. Added cutoff test bisects to adjacent
scales at ranks23/28 and compares all three policies at nine neighbors each (54 probes)
exactly, including contribution counts. No universal numerical equivalence claim.

Initial CMake used keyword linking on a plain-link target; corrected consistently and
made the baseline fixture available with Build_Benchmarks OFF. Initial NCA fixture
requested two indices although its domain has one; corrected range using reported count
and resets G for each scenario. Failed/final logs retained. No implementation test
failure was waived. All GPU verification sequential; independent build trees can build
concurrently. User files preserved; no install/push/supervisor/Arrhenius changes.

Freeze includes evaluator sources and checksums; generators now read those sources.
DeviceOps/DlrOps/model helpers still link from production and require their own audit
if changed. New production choices use ordinary shared C++ code, not generated code.
Consolidated binding/Solver preferences/archive/MPI propagation of storage policy remains
next. No new whole-suite, complete physical N14, target integration or speedup claim.
GH200-class target; local A4000 qualifies ported glue and memory/error behavior.

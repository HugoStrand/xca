# RTX A4000 production qualification

Source82ec8cc: six-tensor NCA admission and all retained experimental targets built.
Binding regenerationOFF, no generated source changes. CPU81/81 passes50.52s,
nativeCUDA91/91 passes247.15s, library-onlyCUDA82/82 passes156.27s. Production
CTest excludes ^experiments/; includes existing dense/legacy/block-sparse C++ and
all32 registered Python entries. CPU disabled GPU cases are not device evidence.

Imported capabilities: CPU cuda/small/ordered/mapped=false; nativeCUDA alltrue;
library-only cuda=true, small/ordered/mapped=false. Library-only executes CUDA
reference controls and asserts unsupported native requests. Block-sparse Python
runs7methods on both CUDA configurations with no skips; CPU skips6GPU methods.
The14-method backend suite skips7 onCPU and3 on library-only (see transcripts).

Two MPI ranks each pass7 block-sparse methods in119.179s; launcher0. Includes all
observables, archives, dynamic channels, collective initialization/count/evaluation
errors, host-placement recovery and NCA/OCA bounds. Inherited X11 authorization
diagnostics do not prevent successful unittest/launcher results.

CPU tests parallel2, CUDA/library-only parallel1. MPI runs between GPU suites with
no competing local GPU work. No install/push/Arrhenius allocation. These are
correctness runs, not timing comparisons. GH200 remains performance target;
physical14-mode and fullGH200 solver qualification remain open. Full temporary
logs are hashed; focused transcripts/test lists/summaries are retained here.

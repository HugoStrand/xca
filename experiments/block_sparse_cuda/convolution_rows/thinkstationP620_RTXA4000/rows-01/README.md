# Row convolution: RTX A4000 screen

Source 59d7cda; commands, binary hash and return codes in manifest.json. Eight
rectangular shape/row-cap cases, full convolution and row chunks, three warm samples
per route (48 rows). Timings include packing, allocation, execution, scatter and sync;
full reference reuses scratch, row chunks allocate temporary storage. Both routes
repeat exactly on their own outputs and pass finite/CPU/full-reference screens.

Row chunks are slower in all eight local cases. At 210x210x252, cap16 reduces explicit
workspace from 64,947,284 to 22,987,604 bytes, with median latency 32.291 to 47.561 ms.
At 252x210x210, cap16 reduces it to 19,520,084 bytes, with 31.766 to 54.016 ms.
Caller input/output and opaque library allocations are additional; the transformed
right input remains fully resident. This is a capacity tradeoff, not a speedup.

Three correctness controls pass in native and native-disabled CUDA builds (.64 s
for each CTest). Memcheck passes all three, zero errors/leaks. CPU configuration omits
the experiment. See qualification logs. Twelve threshold-neighborhood probes agree
on keep/drop, but caps1/3 differ from full-reference matrices by up to 3.701e-32 near
the 1e-16 threshold. This is no universal cutoff guarantee or production eligibility.
Only output rows are chunked: the GEMM inner dimension is never tiled.

Keep private until composed evaluator liveness, complete-output admission, alias/error
recovery, adversarial cutoff decisions and target-hardware qualification are resolved.
No production policy changes, full N14 call or GH200 row-convolution claim. GH200-class
accelerators are the target; this RTX A4000 campaign only screens the mechanism.

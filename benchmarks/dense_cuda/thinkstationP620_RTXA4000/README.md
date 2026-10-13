# ThinkStation P620 / RTX A4000 evidence

Machine: ts, Lenovo ThinkStation P620, AMD Threadripper PRO 3945WX,
NVIDIA RTX A4000 (16 GB, compute capability 8.6). Recorded CUDA 13.4 /
driver 580.178.04; individual manifests identify the exact run environment.
GH200-class GPUs are the optimization target and require separate measurements.

`trace_*.csv` contains the local Nsight profiling summaries. `ORIGINAL_README.md` preserves the original run/profiling methodology and findings; its commands describe the layout at the recorded revision.

Historical manifests and raw records are preserved byte-for-byte. Paths in recorded
commands refer to their original revision. Mixed-backend raw datasets remain intact.

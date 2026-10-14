# Production suite refresh after real NCA integration

Source b5a131c (runtime code fbbde67). CPU, native CUDA and library-only CUDA
configured and built all targets, including retained experiments/benchmarks. CPU
binding regeneration left tracked sources unchanged. Production CTests exclude
`^experiments/`; each tree includes 32 Python entries. GPU suite stages were serial:
library-only first, native second. CPU tests use two workers.

- CPU: 83/83, 52.38 s. Disabled GPU cases do not qualify device behavior.
- Native CUDA: 93/93, 302.13 s, including optimized NCA/sparse controls and dense baselines.
- Library-only CUDA: 84/84, 159.61 s. Both native options OFF; rejection/skip behavior qualifies portability.

Full configure/build/test logs, raw CTest output, test inventories, compiler/GPU
metadata and cached dependency hashes are retained. These are correctness runs,
not timing benchmarks. The preceding two-rank MPI qualification is in real-nca-python-01;
no runtime code changed since it. Target-cohort recipe syntax/local replay is separate.

RTX A4000 is a development proxy for GH200-class targets. New NCA target qualification,
higher-order optimized scalar numerical policy and larger convolution capacity/full
fourteen-mode execution remain open. Existing bounded 10-mode and 14-mode primitive/
shape controls do not imply full-system qualification. No cluster hours, install,
push or changes to unrelated user work. Initial manifest assembly used a lowercase
GTest cache path and stopped; corrected to the actual GTest_src before recording.

Archived text uses LF and strips trailing whitespace for repository checks; JSON is reindented without changing payload. Original hashes for normalized artifacts are in manifest.json. Runner output and verification results are unchanged.

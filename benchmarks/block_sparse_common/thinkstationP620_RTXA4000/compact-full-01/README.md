# Compact DLR complete-call control

Implementation `13ab3b0`. All 35 isolated processes / 175 samples passed, including
reference, two bounded kappa caches, mapped weighting and mapped/compact DLR.
Warm output norms, control-error metadata and discovery counts agree between the
mapped and compact variants; repeated results are exact. The private full evaluator
changes only DLR operations and reserves five rather than six DLR rank buffers.
Original caller/function/operator envelopes and production routing are retained.

[Manifest](manifest.json) pins sources/headers/binaries/commands; [samples](samples.csv)
and [summary](summary.csv) retain comparisons. Raw GPU files are in the matching
CUDA benchmark hardware folder. Each route has five samples per case in one process;
no matched CPU speedup or independent-process confidence interval is asserted.

| Modes / partition / order | Mapped ms | Compact ms | Change |
| --- | ---: | ---: | ---: |
| 4 / number / 1 | 4.459 | 4.520 | +1.4% |
| 4 / number / 2 | 6.139 | 6.087 | -0.8% |
| 4 / mixed / 2 | 11.459 | 11.171 | -2.5% |
| 4 / occupation / 2 | 15.177 | 15.075 | -0.7% |
| 6 / number / 2 | 12.781 | 12.288 | -3.9% |
| 8 / number / 1 | 29.958 | 30.375 | +1.4% |
| 10 / number / 1 | 188.585 | 185.984 | -1.4% |

Changes range from a 3.9% improvement to a 1.4% regression, below a robust speedup
promotion gate. The intended benefit is memory scheduling. NCA performs no DLR
convolution: its lower reported bound is conservative reserve refinement, not
measured released working storage. The ten-mode declared bound drops from
473,446,400 to 450,076,928 bytes. Extrapolated N14/r26 global reserve saves one
maximum tile (4.563 GiB); it still does not prove full fourteen-mode admission.

The [primitive study](../../../../experiments/block_sparse_cuda/compact_dlr/thinkstationP620_RTXA4000/compact-01/README.md)
separates retained storage from the later solve's boundary peak and checks aliasing.
All builds/tests/sanitizers completed before timing. Eight inherited complete
observable/API tests passed with the compact reference policy; mapped complete-call
screening here is additional evidence. No production/Python default changes, no
full N14 evaluation or GH200 capability claim. Next qualify scheduling-specific
admission rather than reserving inactive convolution scratch for NCA, and retain
exact numerical/ownership controls before promoting any compact policy.

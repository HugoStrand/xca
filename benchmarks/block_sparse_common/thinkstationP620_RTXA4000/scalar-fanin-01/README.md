# Output-owned ordered GPU accumulation

Code `e52609d`. All 30 isolated processes / 82 samples passed numerical screening
and exact replay; discovery counts match the retained serial candidate. Data and
hashes: [manifest.json](manifest.json), [samples.csv](samples.csv),
[summary.csv](summary.csv). CPU and GPU were freshly paired; serial GPU values
come from the separate clean scalar-matched-01 run. No build/test jobs overlapped.

The new route keeps the same per-lane host cutoff scan, then groups surviving
lanes by output. A thread owns each output/rank entry and adds contributions in
original lane order, including the prior output value. No atomics/tree reduction
or path sum reassociation is used. Metadata preparation/upload and stream completion
are charged to full calls. Serial accumulation remains available as a control.

| Modes / partition / observable | CPU matched ms | GPU serial ms (prior) | GPU ordered kernel ms | Serial / kernel |
| --- | ---: | ---: | ---: | ---: |
| 4 / occupation / se | 2.708 | 3.429 | 3.327 | 1.031 |
| 4 / number / se | 1.271 | 10.807 | 10.940 | 0.988 |
| 4 / mixed / se | 1.538 | 14.067 | 14.072 | 1.000 |
| 4 / occupation / spgf | 1.298 | 12.557 | 12.335 | 1.018 |
| 4 / occupation / correlator | 1.371 | 10.584 | 10.519 | 1.006 |
| 4 / number / spgf | 1.703 | 23.751 | 23.388 | 1.015 |
| 4 / number / correlator | 1.509 | 9.830 | 9.806 | 1.002 |
| 4 / mixed / spgf | 1.354 | 18.260 | 18.565 | 0.984 |
| 4 / mixed / correlator | 1.488 | 11.855 | 11.854 | 1.000 |
| 6 / occupation / se | 0.031 | 1.981 | 1.977 | 1.002 |
| 6 / number / se | 20.415 | 31.187 | 31.243 | 0.998 |
| 6 / mixed / se | 11.444 | 42.219 | 41.681 | 1.013 |
| 8 / number / se | 248.203 | 92.570 | 92.394 | 1.002 |
| 10 / number / se | 6949.764 | 701.660 | 699.485 | 1.003 |
| 10 / occupation / se | 22.908 | 89.399 | 56.143 | 1.592 |

The accumulation change improves eligible scalar self-energy but the sampled large
scalar point still trails matched CPU. GPU-losing and zero-survival rows are retained.
CPU batching applies only to eligible scalar SE; other CPU rows use unchanged CPU,
and the GPU retains general execution there. Do not read all rows as a batching
comparison or choose production/GH200 defaults from this screen.

Baseline/candidate tests pass 12/13 cases; the extra case uses large cancelling
terms, interleaved owners and existing output to catch reordered sums. Focused
memory/leak checking passes five cases with zero errors/leaked bytes. A separate
three-sample ten-mode smoke is retained in temporary logs; this campaign's ten-mode
points have one sample. No whole Solver speedup, target hardware or N=14 full
qualification follows. See [methodology](../../../../experiments/block_sparse_cuda/scalar_paths/README.md).

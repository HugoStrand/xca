# Six-tensor NCA admission

Production NCA subtracts one additional maximum rank tile from its former admission
reserve. T+out+kap+weighted+mu+term bound the reference special fallback; mapped
execution uses fewer. There are no NCA convolution edges and retained DLR scratch
is released before admission. CSR/workspace and weights have separate reserves;
previous sparse cache is released before replacement. Generic/convolution/correlator
bounds, arithmetic and buffer allocation are unchanged. This tightens a conservative
reserve; it is not an observed reduction in allocated storage or a timing speedup.

Five controls/44GoogleTests pass64.03s, including historical NCA admission (its old
seven-tensor reserve is preserved), scalar_streams, production admission/observables
and live planner comparisons. Two added liveness controls pass12.05s:24 dynamic
small-system routes across number/mixed/occupation, Library/Ordered sparse backend,
Reference/Mapped and both output placements, plus two bounded10-mode number-sector
routes. Number/mixed retarded operators explicitly fail mapping eligibility, checking
the six-tensor fallback. Tight budget passes; bound-minus-one rejects before work;
changed inputs remain active and match a generous-budget evaluator exactly. Full
G/output returned even for the one-index ten-mode window. OCA rejection also checked.
Focused24-route dynamic memcheck passes55.940s, zero errors/leaks. Two CPU-only
metadata controls pass. No broad full-suite or complete14-mode execution claim.

Observed10-mode reference deltas: resident274,726,912bytes within309,795,936 bound;
host-sector226,492,416 within265,175,200. Tile23,369,472bytes. cudaMemGetInfo sampling
includes allocator/context effects and may miss peaks between checkpoints; tests
allow64MiB overhead. Source liveness and exact-bound admission remain separate from
sampled physical accounting. Sanitized peak rows are excluded from observed-peaks.csv.

N14 host-sector Reference NCA estimate at rank26:52.081→47.517GiB; at rank64:
128.154→116.921GiB. Both fit a declared120GiB payload budget. Convolution remains
74.898/184.318GiB, host output15.543/38.258GiB. Old/new sixteen metadata rows assert
exactly one tile NCA difference and unchanged convolution/correlator/host output.
Rank is diagnostic, not physical beta/window/accuracy qualification. Additional
caller input/model/atom_diag/host temporaries, allocator/library/context/headroom
and actual free HBM still matter. Default512MiB host-output budget would reject
these full14-mode outputs; the diagnostic explicitly declares64GiB. GH200 is target,
no new target allocation or full14-mode numerical/capacity qualification occurred.

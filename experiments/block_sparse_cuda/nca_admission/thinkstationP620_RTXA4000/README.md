# NCA admission qualification

Ten tests pass in 8.01 s, including eight inherited observable/API cases. Two focused
budget/transition/release cases pass memcheck/full leaks in 7.40 s with zero errors
and zero leaked bytes. Timings are correctness checks, not performance measurements.
Reference and compiled mapped policies both pass tight-budget acceptance and exact
production comparisons. The source/commands and limits are in [methodology](../README.md).

[Analytic envelopes](analytic_envelopes.csv) use binomial dimensions only, without
atom_diag or N=14 evaluation. They omit fixed tables, operator reserve, weights,
mapped metadata, host and opaque allocations. N14/r26 variable reservation falls
from 90.409 GiB to 63.029 GiB by omitting six inactive scratch tiles. Inputs/outputs
still require the complete function envelope. This is not a fitting full N=14 call
or target hardware qualification. GH200 remains the performance target.

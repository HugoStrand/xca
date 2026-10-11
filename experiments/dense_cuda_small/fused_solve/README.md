# Cached-LU solve follow-up probe

Isolated follow-up to E4 after its trace shifted the bottleneck to pivoted solves.
Ordinary SE only. XCA_SOLVE_MODE=1 (default) replaces each cached LU OP_T solve
and layout packing with a per-RHS CUDA thread;0 retains the E4 library solve.
Existing cached factors/pivots are used unchanged, with U^T forward substitution,
unit-diagonal L^T backward substitution and reverse pivots. No conjugation, explicit
inverse, factorization, mixed precision or tolerance change. Rank<=40 bounds private
scratch. XCA_FUSED_MODE=2 enables the already checked N4 products for this combination.

The custom solve is a numerical feasibility probe, numerically rejected by strict parity; see RESULTS.md. Sequential
substitution can alter rounding relative to cuBLAS blocked TRSM; private runtime
indexed scratch can spill. Do not promote based on speed without numerical gates.
Standalone CMake mirrors fused_small and adds solve_micro (21 alternating cached
primitive timings at ranks16/26/36; coefficient and reconstruction diagnostics).
Strict CTests are not weakened or marked WILL_FAIL. Library control passes; custom mode is not recommended. Results and reproduction
commands are in RESULTS.md.

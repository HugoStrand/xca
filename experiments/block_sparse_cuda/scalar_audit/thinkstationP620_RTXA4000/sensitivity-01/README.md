# Observable sensitivity and cancellation

Three fixtures, analytic directional/bilinear checks and224/320binary-bit convergence
pass with1e-25 criterion. Initial160-bit attempt failed precision convergence on the
oscillating fixture; increase precision, not tolerance. All42 previous GPU stage rows
remain exactly unchanged. Added calculations are host-only; no device algorithm,
allocation ownership, selector or default changed. No new timing/memcheck claim.

| Input | Node Jacobian f | Absolute expanded scale | Ideal coefficient-change effect |
| --- | ---: | ---: | ---: |
| smooth/r23 | 1.15 | 2.16 | 2.04e-16 |
| oscillating/r36 | 2.19e7 | 2.60e24 | .0718 |
| complex/r47 | 4.06e9 | 4.00e25 | 36.2 |

Coefficient Jacobian row norms remain~2–3; node sensitivity reflects composition
with the double cached-LU inverse. Raw matrix conditions alone would hide the smooth
case's modest output sensitivity. Conversely, small coefficient perturbations can
trigger much larger executed GPU differences through roundoff on enormous intermediate
terms. The ideal coefficient-change effects are vastly smaller than the previous
executed general-convolution changes3.94e7/3.24e8. Those executed changes include
input-sensitive floating arithmetic, not just the ideal Jacobian response.

Norms here use componentwise complex l1, unlike prior Euclidean output maxima; they
are not directly equal measures. High precision treats the sampled double tables/
coefficients as inputs. It does not prove physical DLR accuracy, invert exact matrices,
or certify cuSOLVER/cuBLAS error. The supplied inverse is still double approximate.
Precision convergence verifies these diagnostics, not an optimized replay policy.

Next error model must include both observable sensitivity and absolute intermediate
roundoff scale. Ill-conditioned/cancellation-heavy inputs need general replay when
uncertainty cannot be bounded. GH200 remains target; no new Arrhenius GPU time used.

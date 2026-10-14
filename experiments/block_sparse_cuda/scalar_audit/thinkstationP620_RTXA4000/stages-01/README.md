# DLR stage isolation

Final42rows: rank23 smooth physical small-system inputs, rank36 oscillating nodes,
rank47 complex nodes. All values finite. Broadcast solve coefficients are exactly
identical across512lanes before same-coefficient comparisons. Inputs unit scale;
these output magnitudes are not the preceding threshold-scaled diagram samples.

| Pattern | Estimated condition | Max f coefficient | Narrow f residual | Wide f residual |
| --- | ---: | ---: | ---: | ---: |
| smooth/r23 | 1.12e11 | 1.09 | 2.88e-16 | 2.00e-16 |
| oscillating/r36 | 1.02e13 | 4.70e11 | 9.69e-5 | 7.80e-5 |
| complex/r47 | 2.02e15 | 2.76e12 | 1.09e-3 | 9.23e-4 |

One/512-column f solves differ9.98e-8/8090.5/1.86e7. Smooth convolution remains
within2.78e-16 despite coefficient differences. Difficult convolution differences
are5.90e7/1.15e8. Injecting wide coefficients into the narrow general convolution
isolates solve effects (3.94e7/3.24e8). Even same coefficients leave wide-pack
arithmetic differences (2.52e7/3.04e8). Thus solve residual alone cannot certify
observable/cutoff agreement, nor can a rank threshold inferred from these cases.

Replacing literal complex products/additions with general BLAS primitives makes
narrow pack/general convolution exactly equal on all three fixtures. Independent
scalar BLAS products with identical coefficients are exactly equal across widths.
Wide rank transforms still differ: complex Hf/Hg errors6.15e-5/3.08e-5 on ~1e11
magnitudes. Cancellation amplifies those small relative differences; wide BLAS-
product variant still differs (complex3.35e8). Product replacement alone is not a
safe optimized evaluator. No performance evidence or production selector changes.

Residuals use host long double (64mantissa bits) against sampled double C. Condition
is double cached-LU inverse row norms; neither is interval/exact certificate. No
claims of mathematical convolution accuracy on arbitrary node patterns. Comparisons
use executed general CUDA as reference, independently of prior diagram audit.

Twenty retained streams controls pass46.26s after factoring shared pack coefficient
execution; order/math unchanged. Final stage memcheck zeroerrors/leakedbytes. No
competing GPU work, target allocation, install/push. Source/binary hashes and final
raw rows retained; intermediate27/33row outputs lack capture hashes and are labeled
exploration. GH200 remains target; local A4000 stage effects are numerical evidence.
Next certificate must cover solve width, product arithmetic and rank-transform
cancellation, with general replay when uncertainty is not bounded tightly enough.

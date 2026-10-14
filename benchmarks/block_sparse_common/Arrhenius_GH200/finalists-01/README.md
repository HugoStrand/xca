# GH200 fresh-process production finalists

Arrhenius n123, one GH200, Arm/GCC14.3, CUDA12.9; job3853285 completed0:0.
Pinned source d9fa8a2/runtime85ecb2d, recipe b052919. The manifest retains exact
commands and runner/binary hashes. Single OpenMP/BLAS thread, rank23, beta2,
omega_max20, eps1e-10, one declared topology/index per case. Six fixed scientific
cases and seven controls, interleaved within each repetition. No N10/N14 repeats.

All126processes passed, with126distinct PIDs, three fresh processes per group and
three synchronized C++ call samples per process (378total). Every signal is nonzero;
maximum absolute error1.333e-14, relative error5.708e-15. validation.json records the
independent archive checks. summary.csv reports the median of three process medians,
minimum/maximum process medians and MAD; samples.csv retains every sample, setup,
error and memory field. These are whole evaluator-call latencies, excluding evaluator
construction; setup is reported separately. They are not full Solver throughputs.

Median of process medians (ms):

| Case | CPU | CPU constants | CPU hoisted | Library | Ordered | Mapped | Real-NCA policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar-n4-nca | 0.211610 | 0.171003 | 0.211291 | 27.804344 | 9.748176 | 9.796495 | 0.200155 |
| scalar-n8-nca | 29.699492 | 3.533406 | 29.023254 | 938.382152 | 328.916908 | 296.728270 | 1.352474 |
| scalar-oca-fallback | 1.205694 | 1.673073 | 0.475155 | 27.723930 | 14.191540 | 14.374511 | 14.193588 |
| number-n6-oca | 17.797423 | 17.731890 | 5.617443 | 83.115989 | 29.604615 | 10.659607 | 29.515881 |
| number-n8-nca | 92.509712 | 92.542640 | 92.229655 | 193.268089 | 69.052444 | 13.268238 | 66.883481 |
| mixed-retarded | 7.914947 | 7.888357 | 4.327560 | 42.956467 | 25.842640 | 25.138884 | 24.791341 |

CPU constants is the scalar-matched preparation control; outside its scalar domain
it uses general CPU evaluation. CPU hoisted retains the transform-hoisted experiment.
Mapped uses the actual production Ordered/Mapped evaluator, not the private prototype.
Real-NCA policy is the guarded production real batch, with asserted ordered fallback
for noneligible cases. It does not combine mapped mixing with real batching.

For scalar N8 NCA, real batching is2.61x faster than matched CPU constants and243x
faster than ordered CUDA. For scalar N4 the CPU-constants median is lower (.171ms
versus .200ms), though one CPU process median reaches .289ms. Mapped number N8 NCA
is5.20x faster than ordered and6.95x faster than hoisted CPU; mapped N6 OCA is2.78x
faster than ordered but still slower than hoisted CPU (10.660versus5.617ms).
Mixed-retarded and higher-order scalar windows favor CPU controls. These negatives
are retained, along with the mixed-retarded CPU outliers (see process ranges).

Mapped special mixing declines singleton sets and this SE-specific path is not
applied to the correlator case. Timing differences in these fallback cases do not
establish a fusion benefit. Unsupported real-NCA cases retain ordered GPU fallback;
no fallback median regresses by more than5% in this cohort. No runtime defaults or
automatic thresholds change. Three fresh processes establish repeatability only
for these bounded rank23 inputs, not a general crossover, threaded-CPU comparison,
physical high-rank accuracy or full N10/N14 qualification. Target is GH200-class;
local RTX A4000 screening does not set production thresholds.

Raw CPU and CUDA files live in their respective
[CPU](../../../block_sparse_cpu/Arrhenius_GH200/finalists-01/README.md) and
[CUDA](../../../block_sparse_cuda/Arrhenius_GH200/finalists-01/README.md) folders.
Build, four focused CTests, accounting and source identities are in
[qualification records](../../../../experiments/block_sparse_cuda/gh200_qualification/Arrhenius_GH200/finalists-01/README.md).
Original downloaded hashes precede text whitespace normalization. Allocation includes
build/test time:329GPU-seconds; cumulative tracked3356seconds/.932222GPUh against the
shared100GPUh/month ceiling. Other user work is not included. Job is terminal.

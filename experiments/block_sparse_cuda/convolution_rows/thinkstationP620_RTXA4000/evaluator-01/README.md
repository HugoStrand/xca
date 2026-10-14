# Complete row-convolution screen: RTX A4000

Source 21e1100. Binary hashes, commands, caps and process statuses in manifest.json.
Shared complete-call benchmark fixture: number-sector OCA at N4/N6/N8/N10 and N6
mixed-block SPGF, one topology/index, rank23, one OpenMP/BLAS thread. Reference and
row routes use library sparse products and reference special vertices. Three warm
samples per ordinary route; a separate one-sample row allocation diagnostic adds
free-memory samples while temporary buffers are live. Total 35 rows, every process
passes finite/CPU-screening/exact replay controls; ordinary structural counts agree.

| Case / row cap | Reference median ms | Row median ms | Reference bound bytes | Row bound bytes |
| --- | ---: | ---: | ---: | ---: |
| N4 number OCA / 1 | 32.948812 | 36.277596 | 33809184 | 33767232 |
| N6 number OCA / 4 | 103.294809 | 106.710776 | 36045152 | 35603552 |
| N8 number OCA / 8 | 265.177671 | 278.068180 | 64709408 | 58527008 |
| N10 number OCA / 16 | 1115.999539 | 1213.258087 | 450012768 | 363953760 |
| N6 mixed SPGF / 4 | 176.361439 | 182.564274 | 34424624 | 34371632 |

N10 declared bound falls 19.1%, latency rises 8.7%. Sampled global device-memory
changes are in summary.csv; driver caching/granularity makes zero deltas possible.
The separate row-sampled N10 process observes 287309824 bytes vs reference's
343932928 bytes; these are neither continuous peaks nor process-local payloads.
No target-hardware/default/whole Solver inference from three local timing samples.

Both native and native-disabled variants pass ten complete-evaluator controls in
9.62/9.54 s. CPU configure omits the experiment; both native flags OFF in library
tree. Focused memcheck two additional controls passes in 23.47 s, zero errors/leaks.
Inherited controls include dynamic interactions, correlators and high-order ranges;
new controls force cap1, changed G/masks, both output placements, exact-budget
admission/rejection and recovery. Initial observer test incorrectly required a positive
global free-memory delta; removed that unsupported assertion, retaining comparison
and admission controls. Build/test logs preserved, including that failed iteration.

The final complete-call cutoff audit executes 54 probes at ranks23/28 and caps1/3/8.
At rank23/cap1, scale 2.1680390464468913e-05: reference retains one contribution,
candidate zero; error/reference norm both 1.0000000000000003e-16. All other sampled
counts agree. A completed audit process is not semantic acceptance. This concrete
counterexample blocks production promotion despite ordinary tests passing.

Capacity estimates are metadata only; see capacity-method.md. Row chunking leaves
N14/r64 host-output OCA at139.648 GiB, above this GH200's95.58 GiB. Caller tensors
and resident input therefore need further liveness/residency work; row scratch alone
is insufficient. Preserve this failed semantic candidate and baseline; do not add a
heuristic cutoff band. GH200-class target, A4000 local screen. No Arrhenius GPU time
used, no Python/MPI integration or new broad-suite qualification.

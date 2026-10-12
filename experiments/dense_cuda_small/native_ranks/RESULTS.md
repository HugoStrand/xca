# Native rank qualification results

Local RTX A4000 control, CUDA runtime13.4, driver API13.0, cuBLAS13.8.0, cuSOLVER12.3.4. GH200-class cards remain the optimization target and are unqualified locally. No production rank expansion is implied by this report.

The current rank36/N8 screen passes84 backend/case groups. Together with earlier rank26/N4, rank36/N4 and rank26/N8 screens, coverage includes ordinary/paired self-energy, crossing/chain, orders1/3/6/8, counts8/128/129 and strict individual/aggregate CUDA comparisons. Current repeat cohorts use the same generated native snapshot in three fresh processes per rank/dimension, seeds31415/1729/31415, three timed samples per case, crossing, orders6/8, counts128/129. All processes exit0; aggregate candidate/reference errors in the saved repeat summaries are exactly zero.

| Rank | N | Method | Order |128 median range(ms)|129 median range(ms)|CPU speedup range(128)|
|---|---|---|---|---|---|---|
|26|4|se|6|12.633–12.855|16.614–16.738|19.96–20.35×|
|26|4|se|8|17.190–17.386|22.847–23.310|20.85–21.03×|
|26|4|pairs|6|12.711–12.761|16.763–16.959|20.42–20.81×|
|26|4|pairs|8|17.342–17.467|22.895–23.205|20.78–21.01×|
|26|8|se|6|41.342–41.373|45.450–45.487|17.48–17.75×|
|26|8|se|8|57.264–57.386|62.915–63.053|17.61–17.88×|
|26|8|pairs|6|42.056–42.269|46.253–46.417|17.35–17.73×|
|26|8|pairs|8|58.008–58.042|63.749–63.914|17.50–17.89×|
|36|4|se|6|18.805–18.831|24.467–24.472|29.31–29.97×|
|36|4|se|8|25.926–26.096|33.738–33.952|29.84–30.58×|
|36|4|pairs|6|18.911–18.930|24.579–24.725|29.40–33.36×|
|36|4|pairs|8|26.047–26.068|33.900–34.053|29.90–30.52×|
|36|8|se|6|68.133–68.260|73.915–74.016|23.27–23.42×|
|36|8|se|8|94.904–94.945|102.936–102.992|23.38–23.59×|
|36|8|pairs|6|69.191–69.213|75.058–75.084|23.16–23.28×|
|36|8|pairs|8|95.913–96.000|104.031–104.157|23.21–23.38×|

Ranges combine seed/process variation, not confidence intervals. First-call/setup/context costs, owned-storage reports, exact selected indices and high-order CPU diagnostics remain in raw CSV/JSON records. CPU comparisons at high order are diagnostic; no tolerance changed. The retained CUDA evaluator is a correctness reference here and is not timed in these cohorts.

All eight direct reused analytic/dynamic/unitary/index/layout/SPGF/correlator suites against the actual probe pass (10.98s). Wider rank eligibility must await that evidence and varied channel/pole fixtures. Shape2 remains serial library because safe per-lane LU batching lost performance in earlier experiments. Larger powers of two require physical/crossover coverage; compiled kernels are not sufficient qualification.

Production environment admission is now restricted to the measured local hardware/library tuple; focused native/Python/policy tests pass3/3 and the CPU policy check passes. Unknown environments retain native sequential library work; the explicit experiment bypasses admission to collect new hardware evidence, with strict gates and batch-counter assertions. Only GH200-class measurements may justify target defaults.

# Packed production NCA complete-call timings

Local RTX A4000; target remains GH200-class hardware. Three samples for each route/case, 36 rows. All calls pass finite-output, CPU screening and deterministic replay controls; numerical/error/path-count summaries agree across routes. Exact-input tests and memory checks are recorded in production_qualification/thinkstationP620_RTXA4000/real-nca-arena-01. Full commands and binary/source provenance are in manifest.json; unchanged control binaries are explicitly identified.

Median milliseconds (general ordered / retained prototype / production): N4 NCA 9.351 / 1.173 / .211; N8 NCA 293.538 / 2.158 / 1.284; N4 OCA fallback 15.009 / 15.246 / 15.113; N4 number-partition fallback 8.721 / 8.532 / 8.558.

Packing improves production versus its previous campaign (.660 to .211 ms for N4, 8.602 to 1.284 ms for N8). Relative to contemporaneous prototype controls, production is 5.56x / 1.68x faster on N4 / N8 NCA. General ordered controls are 44.4x / 228.6x slower. These are bounded local cases with three samples, not universal crossovers, GH200 evidence or full Solver/topology-sum speedups. Other orders/partitions retain general CUDA and unchanged numerical policies.

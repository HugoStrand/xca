# First production NCA batch complete-call timings

Local RTX A4000, three samples per route/case, 36 rows. Target is GH200-class hardware; these timings do not establish target defaults. Full commands, source commit and binary hashes are in manifest.json. The common runner checks finite output, CPU screening and deterministic replay; signal/error/path-count summaries agree across routes. Exact-input comparisons are in the production qualification tests.

Median milliseconds (general ordered / retained prototype / production): N4 NCA 9.368 / 1.146 / .660; N8 NCA 295.361 / 2.146 / 8.602; N4 OCA fallback 15.203 / 15.065 / 14.993; N4 number-partition fallback 8.511 / 8.570 / 8.470.

The N8 regression relative to the prototype motivates testing packed sector storage. These measurements precede that change and are retained as a control. Three samples are insufficient to infer a universal crossover; CPU preparation differs from experiment controls in earlier campaigns. No Solver/topology-sum or target speedup claim.

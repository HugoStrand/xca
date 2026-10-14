# Local replay of the capped target NCA cohort

Recipe a6ae6ef, production source b5a131c; local RTX A4000, not GH200. Four cases,
four routes, three samples: 48 rows. All processes succeed; each frontend checks
finite output, CPU screening and deterministic replay. Offline GPU norm/error/path
summaries agree. Commands and runner/binary hashes are in manifest.json. CPU constants
is an experimental control with different preparation from the unchanged CPU frontend;
this is not a universal CPU ceiling or crossover claim.

CPU constants / general ordered / prototype / production medians (ms):
N4 NCA .446 / 9.484 / 1.123 / .216; N8 NCA 4.157 / 292.372 / 2.102 / 1.326;
N4 OCA fallback 2.637 / 15.237 / 14.924 / 15.010;
N4 number fallback .275 / 8.550 / 8.658 / 8.534.

This qualifies local runner execution/recording on real binaries. Shell syntax passes
for real_nca.sbatch; remote scheduler validation/submission and target execution remain
pending. The prepared one-GPU job is capped at fifteen minutes (max .25 nominal GPU-hours)
and installs nothing. No cluster hours were consumed.

Archived text uses LF and strips trailing whitespace for repository checks; JSON is reindented without changing payload. Original hashes for normalized artifacts are in manifest.json. Runner output and verification results are unchanged.

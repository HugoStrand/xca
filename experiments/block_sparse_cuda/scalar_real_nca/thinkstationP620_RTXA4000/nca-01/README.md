# Real NCA qualification and complete calls

Eleven candidate controls pass59.07s; twenty retained stream controls pass46.95s.
Four Library/Ordered × Reference/Mapped combinations match general numeric arrays
exactly with real multi-set data,>512paths/tail and changed inputs. Complex G and
model tables, higher orders and host-output placement use general execution.
Adjacent cutoff doubles need no replay; live subnormal node checked. Unsafe-range
fallback and subsequent safe reuse verified. Two focused controls pass memcheck,
zeroerrors/leaks. No universal signed-zero bit-pattern or GH200 claim.

Four cases/three routes/three samples (36rows) pass finite outputs, CPU screening,
exact repeatability, matching GPU error/norm/count summaries. Frontend asserts route
usage. CPU is constants-batched control, not necessarily identical host preparation.

| Case | CPU constants ms | General ordered ms | Real NCA ms |
| --- | ---: | ---: | ---: |
| N4 occupation NCA | .436 | 9.351 | 1.147 |
| N8 occupation NCA | 4.130 | 298.376 | 2.158 |
| N4 occupation OCA fallback | 2.686 | 14.934 | 15.224 |
| N4 number NCA fallback | .273 | 8.560 | 8.690 |

Local fast-path speedups8.15/138.25× versus general; N8 beats CPUcontrol1.91×,
N4 still loses. Three samples establish screening only; no universal crossover.
Fallback overhead~2%/~1.5% here. One bounded indexed call/crossing topology,rank23,
no dynamic channels in timing cohort (semantic tests cover real dynamic multi-sets).
Setup/warm-up excluded. Prototype includes unused packed DLR allocations and
conservative reserves; production extraction can remove them after qualification.

All processes sequential after build/tests/memcheck. No target GPU-hours, install/
push/default change. Source/binary hashes,commands,rawrows and qualificationlogs
retained. GH200 target qualification and consolidated production/Python integration
remain open, alongside higher-order conservative replay and large-sector capacity.

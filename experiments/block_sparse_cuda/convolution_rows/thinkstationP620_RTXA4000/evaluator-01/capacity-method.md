# Metadata-only estimates

Total-number sector dimensions d_q=binomial(N,q), D=max(d_q). Rank tensor bytes
T=16*r*D^2; complete function F=16*r*sum(d_q^2). Fixed payload S comprises five
complex r*r tables, (r+1) integer pivots/status, two r*2 hybridization arrays,
32 MiB sparse operator reserve and one r-vector. Special backend Reference,
NCA backend Reference, complete host-output placement. Current conservative OCA
bound F+13*T+S, private row-cap16 bound F+9*T+5*16*r*min(16,D)*D+S.
These are the production/private evaluator formulas, not measured allocation peaks.
Opaque allocations, free-memory headroom, host atom construction and physical rank
selection are additional. No full N14 atom_diag or GPU call is executed here.
N14/r64 bounds 184.318/139.648 GiB both exceed observed GH200 total 95.58 GiB.
N14/r26 row bound 56.751 GiB does not establish physical accuracy or capacity.

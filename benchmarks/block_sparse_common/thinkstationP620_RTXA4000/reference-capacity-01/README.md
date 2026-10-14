# Production reference capacity preflight

Four host-only CLI cases / sixteen rows: N10/N14, ranks26/64, Reference/Mapped
special policy, resident/host-sector output. Two poles,32MiB sparse reserve,
1×1 correlator endpoints,120GiB device payload budget and64GiB output budget.
Number-sector dimensions are binomial; sum-of-squares independently checked via
C(2N,N). Rank is an assumption, not physical beta/window/accuracy qualification.
No atom_diag, Hamiltonian, propagators or device allocations in the report runner.

| Modes | Rank | Host-sector NCA GiB | Host-sector convolution GiB | Full SE output GiB |
| --- | ---: | ---: | ---: | ---: |
| 10 | 26 | .300 | .423 | .072 |
| 10 | 64 | .692 | .995 | .176 |
| 14 | 26 | 52.081 | 74.898 | 15.543 |
| 14 | 64 | 128.154 | 184.318 | 38.258 |

Table uses Reference special policy. Mapped metadata adds its declared set-width
reserve in raw rows. Host output means the complete SE return for both placements.
Rank64 fourteen-mode NCA already exceeds120GiB despite output streaming. Additional
input/scratch tiling or a tighter qualified liveness bound is needed at this rank;
low-rank feasibility cannot establish a general14-mode solver memory claim.

CPU-only tests cover both sizes/ranks, payload identity, placement invariants,
invalid extents/overflow and BLAS limit rejection. Three CUDA planner controls
compare exact production bounds for both special/output policies and endpoints.
Five focused CUDA CTest entries/40controls pass25.02s, including admission, all
reference observables and adapted kappa2/mapped experiment generators. Kappa16 and
scalar_streams add31 passing controls in the retained test log. No broad full-suite
rerun, end-to-end14-mode evaluation or GH200 capacity measurement is claimed.

Caller input, model/atom_diag, host temporaries, allocator/library bookkeeping,
CUDA context and free-memory headroom are extra. Within-budget flags are static
payload decisions; the production evaluator still checks actual free memory.
Results are machine-independent arithmetic executed locally; hashes/toolchain
context and numerical/admission logs are retained. GH200 remains the target.
No Arrhenius GPU-hours were consumed.

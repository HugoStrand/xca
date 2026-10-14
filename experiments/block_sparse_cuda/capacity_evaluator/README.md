# Bounded composed NCA capacity

Target GH200-class accelerators; the local RTX A4000 screens correctness and guards.
This probe executes the production evaluator on **all** total-number sectors of
N fermionic modes, keeping complete host inputs and returning complete outputs.
Unlike capacity_phase, it exercises input scanning/staging, path enumeration,
sparse products, output-sector transfer and the public compute call together.

The Fock basis and signed annihilation CSR are constructed directly, without a
large atom_diag diagonalization. Every input block is the same scalar time
function times identity. Hybridization is diagonal, with one pole and identical
active-field coefficients. `--fields 1` limits the orbital sum; `--fields N`
includes every mode. This is a deliberately simple algebraic capacity workload,
not a physical Solver solution, generic off-diagonal input qualification, or a
higher-order convolution test. The prior physical/operator tests cover those
independent semantics. Do not infer physical DLR accuracy from this fixture.

A two-state **CPU** evaluator fixes the time dependence and established
parity/reflection convention independently of CUDA. Exact Fock occupation
projectors determine every larger diagonal entry. Every output matrix entry is
checked, including off-diagonal zeros, with error <=1e-12*oracle norm +1e-16.
The oracle must be nonzero and at least one path must contribute. With two
samples, diagonals must also replay exactly; off-diagonals are checked against
zero on both calls. Oracle dependence on the production CPU evaluator is explicit;
this is not an independently derived time-kernel oracle.

Options: `--modes 1..14`, `--fields 1..N` (default 1), `--lambda 10..10000`
(Lambda=beta*omega_max, beta=2, eps=1e-10, nonsymmetric), `--samples 1|2`.
Rank is constructed and reported, never truncated. Lambda4000 gives rank64
locally. `--dry-run` constructs only the rank and small Fock metadata, without
CUDA context or large arrays. `--device-budget-mib` defaults2048, max98304;
`--host-budget-mib` defaults2048, max106496. Declared-budget rejection exits2
before model/GF payload allocation. Invalid arguments/runtime/numerical failure
exit1. Execution above N10 additionally requires `--allow-large`, GH200 hardware
and a Slurm job. Rejections must be retained, not automatically retried at a
larger cap.

Device policy is Library sparse / Reference special / Reference NCA, HostInputs
and HostSectors. Runtime/planner bounds must agree. Complete caller input and
returned output are included in the host preflight, plus two maximum tiles for
construction/download temporaries and512 MiB model/CPU allowance. This is a
probe-specific conservative payload estimate, not an allocator/cgroup guarantee.
Record `/usr/bin/time -v`, the allocation/cgroup memory limit and actual CUDA
free-memory deltas. The production evaluator separately checks device free bytes
with headroom; opaque library allocation failures remain possible.

```bash
block_sparse_capacity_evaluator --modes 10 --fields 10 --lambda 4000 --samples 2
block_sparse_capacity_evaluator --modes 14 --lambda 4000 --dry-run
```

At N14/rank64, one full function is41079398400 bytes (38.26 GiB), explicit device
bound84463062276 (78.66 GiB), host bound106818289664 (99.48 GiB). These are
inventory results until an actual admitted target run finishes. Use an isolated
pinned build, reduced controls first, one capped GPU allocation, no Cartesian
sweep, and respect the shared100 GPU-hour/month Arrhenius ceiling. The default
host cap intentionally rejects the full shape.

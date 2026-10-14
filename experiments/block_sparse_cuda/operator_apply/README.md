# Ordered sparse operator applications

Performance target: GH200-class accelerators. This isolated E3 experiment compares
operator routes; it changes no production evaluator or dispatch default.

The fixture constructs sorted Fock states in adjacent total-number sectors at half
filling and applies the first `terms` annihilation fields with exact fermion parity.
One term is a signed partial map; multiple terms have complex weights and represent
a general sparse barred-field pattern. Row degrees include empty rows. No atomic
diagonalization or eigenbasis round trip is needed. N=14 uses the exact
3003-by-3432 adjacent-sector geometry. It is an operator proxy, not a full N=14 model.

Controls on identical complex-double rank functions:

- `ordered_csr`: one thread per output entry, CSR entries in sorted-column order,
  all DLR nodes in one launch, no atomics. Native kernels disable FMA contraction.
- `cusparse_alg3`: the retained production deterministic CSR path, one node at a
  time. It includes descriptor queries/creation and per-node synchronization.
- `cublas_dense`: one complex-double strided batch, shared dense operator with
  zero batch stride. It stores only one matrix, not rank copies.

Right products use a **plain** transpose snapshot for CSR, with complex coefficients
unchanged. Sparse CPU application supplies the numerical reference. Screening is
absolute error <= 1e-13*reference norm+1e-14; all measured outputs must repeat exactly.
This does not independently qualify atom_diag export (covered in evaluator tests).
The custom route is ordered general CSR, not yet a compact signed-map specialization.

Build with CUDA TRIQS and an available/configured CUDA compiler:

```bash
source ~/env/xca-cuda.sh
cmake -S . -B cbuild-cuda -DUpdate_Python_Bindings=OFF
cmake --build cbuild-cuda --target block_sparse_operator_apply --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 cbuild-cuda/benchmarks/experiments/block_sparse_cuda/operator_apply/block_sparse_operator_apply \
 --modes 14 --terms 7 --width 16 --samples 3 --reuse 8
```

CPU-only builds omit this native experiment. Modes are even 2–14, terms 1–modes,
RHS width 1–64, samples 1–5 and reuse 1–32. Rank is the nonsymmetric Lambda=40,
epsilon=1e-10 DLR rank. No timing implies rank independence or a full Solver gain.
`pack_ms` includes host field construction and plain-transpose packing. `setup_ms`
includes route construction/operator upload and dense expansion for cuBLAS. Context,
input/output allocation and initial input upload are excluded. `first_call_ms`
measures the first apply/synchronization, including lazy workspace/first-use costs.
Warm `call_ms` averages `reuse` submitted applications with synchronization; output
copy/checks are outside that timer. Library CSR also synchronizes each node. Bytes
separate retained operator payload from queried/reused sparse workspace; these are
explicit-storage counts, not complete opaque/host/process peak qualification.

Tests sweep both directions, sparse map and weighted sums, small/mixed/large-sector
shapes and narrow RHS. Promotion needs actual inclusive evaluator/segment timings,
changed-operator costs and target-hardware qualification. Compact maps, implicit
sums, COO alternatives and wider RHS/layout choices remain separate experiments.

Record the bounded 24-case sweep with:

```bash
source ~/env/xca-cuda.sh
python3 experiments/block_sparse_cuda/operator_apply/run.py \
 --binary cbuild-cuda/benchmarks/experiments/block_sparse_cuda/operator_apply/block_sparse_operator_apply \
 --machine thinkstationP620_RTXA4000 --campaign operator-01
```

The sweep uses 2/4/6/8/10/14 modes, one/all-field sums, RHS width 1/16,
three repetitions and reuse eight. It records 432 rows, exact commands, source/
binary/NDA-library hashes and machine/toolchain metadata, with 120 s subprocess
limits and no overwriting. Failures and partial output are preserved. N=14 uses
bounded RHS slabs rather than complete sector-sized rank functions. Sparse nnz,
maximum row degrees (forward/plain transpose), payload and workspace are reported.

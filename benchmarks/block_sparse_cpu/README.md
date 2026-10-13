# Block-sparse CPU workload inventory

Build with `Build_Benchmarks=ON`, independently of examples/tests:

```bash
source ~/env/xca-cpu.sh
cmake --build cbuild --target block_sparse_inventory --parallel 6
cbuild/benchmarks/block_sparse_cpu/block_sparse_inventory \
  --modes 6 --partition mixed --dynamic --order 3 --count 32 \
  --shapes /tmp/block-sparse-shapes.csv
```

Partitions and physical fixtures are described in
[block_sparse_common](../block_sparse_common/README.md). The explicit `--topology crossing` (default) and `--topology chain` catalogues
avoid enumerating all pairings. `--first` selects a bounded index
window; `--count` defaults to 32 and is capped at 4096. Counts include rejected block
paths. Setup includes atom_diag, sparse model export and the DLR mesh. Trace time is
mean CPU scheduling time after one warm-up, with `--samples` defaulting to 3 (maximum
7); shape-histogram construction is excluded from the timed scheduling loop.

Actual atom_diag construction is limited to 2–10 modes. Large-case memory estimates
are analytic, with explicit rank and no diagonalization, propagator allocation,
path traversal or timing claim:

```bash
cbuild/benchmarks/block_sparse_cpu/block_sparse_inventory \
  --analytic --modes 14 --partition number --rank 26
```

The summary reports owned model storage, CSR array payload and a conservative owned
storage bound. Neither includes atom_diag or allocator bookkeeping. Memory columns
for one function, the largest rank buffer, legacy scratch and initial correlator
side arrays are capacity diagnostics; other outputs, temporaries and library storage
are additional. Future device admission must account for actual buffer lifetimes.

Use `run_inventory.py --binary <executable> --output <machine-directory>` to record
bounded cases, shape CSVs, failures, source hashes and machine metadata. System-specific
measurements belong in hardware-named subdirectories; generic definitions stay here.

# Isolated small-matrix CUDA experiments

The production evaluator is the retained baseline. `baseline/source_manifest.json`
records its revision and23 protected source hashes; `baseline/verify.py` rejects
changes to them. Benchmark runs enforce this check automatically.

Read [the plan](../../plan_dense_cuda_small.md) and
[benchmark instructions](../../benchmarks/dense_cuda_small/README.md).
Each strategy receives its own folder, implementation namespace, opt-in build target,
README and results. Do not put exploratory code in the production evaluator.

Planned folders, created when their experiments start:

- `serial_stream/`: owned stream/handles/status, fewer synchronization points.
- `graphs/`: reusable schedule with separately charged graph setup.
- `index_batch/`: independent index lanes, wide shared-LU RHS/rank transforms.
- `fused_small/`: specialized complex-double4×4 operations and fused stages.
- `cpu_hoist_control/`: optional CPU convolution control.
- Named combination folders only after individual strategies have measured evidence.

A candidate factory plugs into `dense_cuda_bench::run` in the shared harness. Keep
unsupported methods explicit. Archive source and negative results; keep build products
under ignored `cbuild*` directories. Preserve the benchmark fixture/timing contract.
Promotion into production requires a later review; this exploration does not overwrite
or automatically select the baseline implementation.

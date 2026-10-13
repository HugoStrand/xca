# Dense benchmark comparisons

Compare matching summary CSVs from the CPU, cuBLAS, native-small or experimental
benchmarks. The shared fixture key includes selected indices and seed; different
case sets fail. Speedup is reference median divided by candidate median, with both
median absolute deviations reported. Check manifests before comparing hardware,
compiler/library environments or revisions.

```bash
python3 benchmarks/dense_comparison/compare.py reference/summary.csv candidate/summary.csv \
  --reference-backend cpu --candidate-backend cuda_small_b128
python3 benchmarks/dense_comparison/solver_workload.py \
  --output benchmarks/dense_comparison/<hardware>/solver-workload/run-01 --samples 3
```

The Solver workload control runs fit/setup, self-consistency, SPGF and a density
observable serially, checking available CUDA backends against CPU. Use the intended
installed/build-tree Python package and its matching runtime environment.

Store cross-backend reports and derived comparison CSVs under the corresponding
hardware folder. Original samples and manifests stay with their benchmark datasets.
[Workstation reports](thinkstationP620_RTXA4000/README.md) are historical local
measurements; GH200-class performance remains to be measured.

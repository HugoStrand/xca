# Scalar completed-path cutoff replay

Private candidate; production and the earlier scalar constants baseline are retained.
Target hardware is GH200-class accelerators; initial controls run on the local A4000.

Batching changes convolution rounding. The active two-mode, one-zero-pole order-six
chain fixture at index 10 finds adjacent input amplitudes around 0.10775810972137
where the original batched candidate discards a lane retained by the general CUDA
reference. This is a cutoff semantic difference, despite tiny ordinary output error.

This variant retains constants/contiguous storage/ordered fan-in. Completed lanes
within an experimental 1e-26 absolute band around the 1e-16 magnitude cutoff are
recomputed through the evaluator's general CUDA vertex/edge/special sequence. Their
values replace the batched lane before cutoff classification and ordered fan-in.
No CPU evaluation or atomic accumulation is used. Replay copies input views into
owned tensors; that extra function payload and scalar lane enter admission.

The band is a diagnostic policy, not a mathematical error bound for arbitrary
condition numbers, ranks, orders, cancellations or invalid intermediate values.
Broader numerical qualification and a conservative promotion policy remain open.
Fallbacks, changed masks/owners, empty ranges and admission retain the shared tests.
The adjacent-order-six test is enabled only for this variant; it exposed a known
failure in the earlier candidate, which remains a benchmark control.

Build `block_sparse_scalar_cutoff_tests` and `block_sparse_scalar_cutoff_benchmark`
with benchmarks/tests and native block-sparse kernels enabled. CTest entry is
`experiments/block_sparse_cuda/scalar_cutoff`. Compare the benchmark with retained
constants GPU and matched CPU, including inclusive replay overhead. No Python/MPI
selector or production default exposes this candidate.

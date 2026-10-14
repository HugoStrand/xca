# Bounded topology prefix inventory

Target hardware is GH200-class; the local RTX A4000 is a development proxy.
This CPU-only structural probe implements the discovery stage of E8. It does
not execute shared numerical nodes, measure performance, or capture CUDA graphs.

Configure with `Build_Benchmarks=ON`, build target
`block_sparse_topology_reuse_probe`, then run in the configured TRIQS environment:

```sh
python3 experiments/block_sparse_cuda/topology_reuse/run.py \
  --build cbuild --machine YOUR_MACHINE --campaign prefix-01
```

The recorder scans at most 4096 pairings from the real `diag.py` generator,
selecting at most six connected topologies per order. Each independent probe
visits the first two flat indices and at most 4096 surviving block paths.
Orders 2, 3, 4 and 6 cover number, mixed and occupation partitions for four
fermionic modes, with active and periodically masked propagator controls.
These first-topology/first-index selections are biased, not population estimates.
The manifest records the actual selections, caps, source and executable hashes.

Keys encode a fixed model/propagator generation, mesh/layout/beta, starting block,
ordered CSR applications, vertex pole weights, propagator blocks, ordered edge
weights and convolution stages. The probe stops before the special vertex:
neither special sums nor suffixes nor final parity/cutoff/output are inventoried.
All calls use one immutable fixture. Literal generation markers are sufficient
only for this inventory; a real cache must invalidate on every input update.

`cross_topology_keys` counts keys present in two or more distinct topologies.
`distinct_topology_reuses` sums distinct topology appearances minus one and
excludes repetition within a topology. Payload bytes count all unique candidate
intermediates, not an implemented cache budget or a measured allocation peak.
Root copies are reported separately from operator/convolution work.

Before implementing reuse, independently replay candidate shared prefixes and
compare values; then measure bounded cache discovery, retained bytes, misses,
changed-input invalidation and complete-call timings. Graph capture additionally
requires stable storage and capture-safe operations. Production allocations,
synchronization and host cutoff decisions currently prevent direct whole-call
capture. Graph replay alone does not reuse numerical values.

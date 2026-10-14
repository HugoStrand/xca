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

Pass `--replay` to the recorder for numerical qualification. Each occurrence is
recomputed using CSR applications and the retained hoisted CPU convolution,
then compared against a separately advanced dense-operator/original-CPPDLR
reference. Repeated keys must give exactly identical recomputed tensors. The
reference screen is `1e-10 * max(abs(reference)) + 1e-14`; this qualifies key
identity and replay, not universal solver accuracy. Nonfinite tensors fail.
Stored candidate values have an explicit total 64 MiB cap; reaching it fails
the probe. Changed-G and changed-coefficient controls run fresh replay processes
and verify changed contents. They demonstrate the need for invalidation, not an
implemented cross-call cache protocol. No device arithmetic is executed here.

## Complete-call device cache experiment

The CUDA-only `block_sparse_prefix_benchmark` uses a generated private copy of
the production evaluator. Its prefix cache stores post-edge tensors only; the
special vertex, remaining path, cutoff and output accumulation stay unchanged.
Model, coefficients, poles and mesh are immutable per evaluator. Before every
nonempty self-energy call, a byte-for-byte snapshot comparison of the complete
propagator input invalidates the cache on change, including changed zero masks.
The snapshot is limited to 32 MiB; larger inputs clear/disable caching and keep
independent device arithmetic. Old/new snapshot peak payload may reach 64 MiB.

Stored tensor payload plus conservative key/node charges is capped at 8 MiB and
4096 entries. Misses beyond the cap compute independently. The admission envelope
charges the full cap; existing device payload is accounted separately from host
metadata. Correlator admission also charges retained cache storage. An explicit
clear synchronizes the owner before releasing entries; device errors propagate.
Metadata key discovery, snapshot comparison, misses, cache copies and input/output
transfers are inside complete-call timing. External cold-cache cleanup and evaluator
construction are excluded. Changed-G invalidation/cleanup occurs inside timing.

```sh
cmake --build cbuild-cuda --target block_sparse_prefix_benchmark block_sparse_prefix_tests
python3 experiments/block_sparse_cuda/topology_reuse/run_cache.py \
  --build cbuild-cuda --machine YOUR_MACHINE --cohorts PATH_TO_REPLAY_COHORTS
```

Compare independent Library/Reference and Ordered/Mapped self-energy calls with cached calls over the
same topologies and two flat indices per topology, three samples per phase. Cold
cohorts populate the cache; repeated cohorts retain it; changed-G cohorts test
automatic numerical invalidation before every sample. Every per-topology complete output must agree
exactly. This retains topology summation order rather than factoring any sums.
The existing eight production observable/control tests are also run against the
private evaluator, including changed masks, dynamics and correlator behavior.
This is an experiment, not a production cache policy or a graph implementation.

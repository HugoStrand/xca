# Shared dense benchmark fixtures

`n4_fixture.hpp` provides the physical models, crossing topology, and deterministic
bounded index samples used by the dense CPU batching and few-orbital evaluator
benchmarks. It contains no machine-specific timing data.

The original `model(high_rank)` is the unchanged two-fermion model (Hilbert-space
N=4) used by the batching benchmark. The overload `model(high_rank, N)` also
supports N=2,8,16, with one to four fermion modes and one retarded density
operator per mode. Models with multiple modes include complex nearest-neighbour
hopping, static density interactions and mixed retarded coefficients. Both
entry points select DLR meshes giving ranks 13 or 103 on the recorded
installation. Sampling remains deterministic with seed 1789 and at most 63
backbones.

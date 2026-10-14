# Complete self-energy output sectors

Target hardware is GH200-class; the RTX A4000 is a development proxy. This private
E7 experiment keeps all returned self-energy sectors, but only one output sector
is resident on the device. Input propagators remain fully resident. This is output
sector streaming, not spatial/contraction tiling or a full large-system solution.

The original path visitor and GPU arithmetic remain unchanged. Before an output
sector switch, the current device sector is copied to its host block and released;
the next sector resumes from its stored host values. Accumulations retain their
per-sector path order and run on the GPU. There is no CPU summation or factoring of
topology terms. Empty/unvisited sectors remain zero with the full mesh/shape.

Self-energy device admission replaces one complete output function with one maximum
rank tile. Correlators/SPGF keep their production implementation and bounds. The
required full host output payload is checked against a private configurable budget
(default 512 MiB) before allocation/upload, including empty index ranges. Rejection
leaves later calls usable. This payload budget excludes caller input, GF metadata,
allocator overhead and a transient largest-sector download buffer; it is not a
process RSS cap. Final output ownership moves into block_gf without copying all
sectors again. Host allocation/device errors propagate without CPU evaluator replay.

Build target `block_sparse_output_sector_tests` and run CTest
`experiments/block_sparse_cuda/output_sectors`. Eight inherited observable/API
controls plus complete-output budget/recovery controls compare number, mixed and
occupation partitions, changed inputs/masks, empty calls and returned full sectors.
No performance or N=14 full-system qualification is implied by these small controls.
Input streaming, slabs, transfer costs, host/opaque peak measurements and target
hardware remain separate gates before production adoption.

Build `block_sparse_output_sector_benchmark`, then run `run.py --machine MACHINE`.
The complete-call comparison has three routes: a copying resident control, production
resident output with an explicit vector move at return, and streamed sectors with that same
move. All outputs and accepted/contributing counts must match exactly. Route order
rotates across three warm/changed-G samples. First-call rows share an initialized
CUDA context and are not independent process-cold measurements. Setup is reported
separately; G changes before each changed-input sample. Inputs/outputs and sector
transfers are inside timing. No MPI or full Solver timing is included.

Memory rows use the evaluator's largest sampled drop in global free device memory
relative to call entry. They include allocator/other-process effects and can be zero
for tiny workloads. They are not continuous, process-private peaks. Host RSS is the
cumulative peak of the process holding all compared evaluators/results, so it cannot
be used to compare per-route host peaks. Bounds report explicit conservative storage,
with the streamed difference exactly one full output minus one maximum rank tile.
The ownership control prevents attributing return-copy savings to sector streaming.
The generator preserves both copy and move variants as production evolves, so
promotion of the move does not silently turn the copying control into another move.

The production C++ evaluator now has explicit `OutputPlacement::HostSectors` with
the same ordering and complete-output contract. Resident remains its default;
Python/MPI selection is pending. `resident_header.hpp.in` and `resident_source.cpp.in`
freeze the pre-placement evaluator used by this experiment, keeping its original
copy/move/stream comparisons independent of future production placement edits.
`block_sparse/cuda_host_output` runs all eight reusable observable controls against
production with HostSectors selected. The original private controls remain separate.

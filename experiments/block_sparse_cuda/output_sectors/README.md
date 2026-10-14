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

# Shared and block-sparse backbone boundaries

Follow-up to the adversarial review of plan_backbone_rename.md, authorized
2026-10-16 on dev_backbone. Keep code and documentation commits separate.

Completed 2026-10-16. Review the code and corresponding log checkpoints in order:

| Step | Code | Documentation | Main change |
| --- | --- | --- | --- |
| 1 | `a98491c` | `a7738d7` | Share backbone geometry. |
| 2 | `46f1607` | `ecefcf6` | Own common backbone prescriptions and separate global factors. |
| 3 | `6341b7f` | This documentation commit | Separate host discovery and CPU execution choices from records. |

Final validation: 74/74 CPU CTest entries, three-rank MPI range selection,
benchmark target compilation, and standalone record-header compilation passed.

The review found shared geometry inside block-specific paths, an enum mixing
correlator geometry with CPU scheduling, and block record headers importing host
preparation dependencies. The correction has three stages:

1. Share backbone ranges between dense and block-sparse evaluation.
2. Give Diagram a shared Backbone owner, separate whole-diagram factors, and
   represent the final correlator propagator edge explicitly.
3. Use ranges for block discovery, separate CPU forward/backward execution, and
   isolate plain block records from host discovery declarations.

No class hierarchy, generic scheduler, GPU implementation or timing experiment.
Preserve arithmetic order, public Python compatibility and unrelated worktree
changes. Future GPU targets remain high-end devices such as GH200.

## Stage 1 — shared geometry (`a98491c`)

Added diagram::BackboneRange with explicit half-open vertex/edge bounds and
validation of an alternating chain. Block-sparse BackbonePath now composes the
range and adds only sector metadata. Both dense evaluation and block discovery
obtain the two correlator ranges from the same common helper. A range may contain
one edge and no vertices; it may include either outer operator. Invalid gaps
between the vertex and edge intervals are rejected.

The CorrelatorSegment enum temporarily remains as an execution selector; stage 3
removes it after the backbone owner and full edge extent are explicit.

Validation: configured and built the CPU tree; six focused shared geometry,
path ownership, prepared evaluator and all-evaluator correlator tests passed
(3.71 s). Generated wrappers are unchanged. Logs:
`/tmp/xca-boundaries-step1-{configure,build,test}.log`.

## Stage 2 — shared backbone owner (`46f1607`)

Diagram now owns diagram::Backbone, containing operator vertices and local
vertex/edge kernel placement. TauKernelPrefactor describes the whole-diagram
scalar factor separately; signs also remain on Diagram. Kernel binding takes
both recipes explicitly and preserves its multiplication order.

Correlator backbones contain all 2m propagator edges, including the trailing edge
to beta with no scalar factors. Self-energy backbones have 2m-1 edges. A block
binder checks ranges against this actual extent instead of accepting any edge
below 2m. The shared owner has no block dimensions, numerical matrices or CPU
execution choices.

Validation: full CPU build passed; eight focused indexing, parity, kernel,
geometry and evaluator tests passed (3.75 s). New checks cover independent
backbone copies, the two full extents and identity binding of the trailing edge
at first and second order. Python wrappers remain unchanged. Logs:
`/tmp/xca-boundaries-step2-{build,test}.log`.

## Stage 3 — discovery and execution boundaries (`6341b7f`)

Removed CorrelatorSegment. Shared correlator geometry supplies two BackboneRange
values. block_sparse::visit_backbone_paths accepts a range with selected operator
vertices, including interior intervals that are neither full correlator half.
Unresolved external endpoints require their own completion discovery. Tests cover
an interior vertex/edge range, an interior edge alone, invalid extents, and the
rejection of a correlator-only edge when binding self-energy operands.

The block_sparse::diagram headers now contain only BackbonePath,
SelfEnergyDiagram and CorrelatorDiagram records. Host discovery lives in
block_sparse/backbone_paths.*, self_energy_paths.* and correlator_diagrams.*,
with functions in block_sparse. Bind headers explicitly include their numerical
dependencies. The compatible path pairs and external-operator choices remain
in block-specific CorrelatorDiagram: common ranges cannot describe sector closure.

CPU forward/backward traversal consumes backbone operands and kernel views.
Boundary dimensions no longer come from a hidden test against the special
correlator vertex. The correlator caller explicitly applies whole-diagram factors
and reflects the backward result. Private traversal routines handle edge-ended
intervals; self-energy endpoint insertion retains its existing separate sequence.

The first full regression run exposed a strided-input error after moving
reflection to the caller: cppdlr::reflect reshapes its input and cannot safely
consume a small active block view into the larger U scratch array. The caller now
materializes that view before reflection. Existing unequal-block, analytic,
partition and MPI comparisons caught the regression; all pass after correction.
No tolerance was relaxed. This requirement is documented at the call site.

Final validation:

- Reconfigured and built the full CPU tree with bindings regenerated.
- All 74 CTest entries passed in 57.93 s, with OMP_NUM_THREADS=1,
  OPENBLAS_NUM_THREADS=1 and CTest parallelism 2.
- Three local MPI ranks passed test/python/range_selection.py. The launcher
  emitted the usual display-authorization notices and exited zero.
- backbone_cpu explicitly compiled and linked; no timing experiment was run.
- A standalone translation unit including the shared Diagram and all three
  block description headers compiled. Its include trace contains no BlockOp,
  OperatorSupportTable construction, or atom_diag headers.
- The Python tree, including generated wrappers, is unchanged from `6a7c792`.
- git diff --check passed. Unrelated initial worktree changes were preserved.

Commands after `source ~/env/xca-cpu.sh`:

```bash
(cd cbuild && bash ../cmake.sh)
cmake --build cbuild --parallel 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild --parallel 2 --output-on-failure
cmake --build cbuild --target backbone_cpu --parallel 6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/cbuild/python:$PWD/cbuild/deps/adapol_src/src:$PWD/cbuild/deps/pyed_src:$PYTHONPATH"
mpiexec -n 3 python3 test/python/range_selection.py
```

Logs: `/tmp/xca-boundaries-step3-{configure,build}.log`,
`/tmp/xca-boundaries-final-test.log`, `/tmp/xca-boundaries-final-mpi.log`,
`/tmp/xca-boundaries-final-benchmark-build.log`, and
`/tmp/xca-boundaries-record-headers.log`. Temporary logs may disappear.

The amended plan section documents the final layout. Remaining limitations are
unchanged: discovery reads host connectivity and masks; exact support must be
rebuilt when operators change; numerical operands borrow live owners; execution
uses single-target paths and evaluator-owned scratch. Initial-sector masks keep
their existing completion policy. These are explicit host implementation choices,
not fields or scheduling requirements of the shared Backbone and BackboneRange.

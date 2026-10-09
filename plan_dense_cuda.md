# Plan: port the dense C++ DiagramEvaluator to CUDA

Status: implemented and verified on 2026-10-10 on `dev_cuda`. The design below records the reviewed plan based on the 2026-10-09 source inspection. The requirement-by-requirement acceptance audit, measured results and limitations are in [benchmarks/dense_cuda/results.md](benchmarks/dense_cuda/results.md); the durable checkpoint is [work_log_dense_cuda.md](work_log_dense_cuda.md).

## Objective and scope

Add `triqs_xca::dense::cuda::DiagramEvaluator` as an additional C++ evaluator. Preserve the dense evaluator's mathematical results, input conventions, topology/flat-index semantics, and host-facing compute interface. Keep its large working tensors on the GPU throughout an evaluation call and use complex-double cuBLAS/cuSOLVER operations. Retain the existing CPU evaluator as the reference and fallback selected by the caller.

The implementation covers self-energy, its by-pairs variant, single-particle Green's functions, one-time correlators, and dynamic interactions already handled by the dense evaluator. Reuse existing C++ comparisons and analytic references. Python bindings, Python solver changes, block-sparse GPU evaluation, MPI changes, fitting, Dyson/self-consistency, automatic CPU/GPU selection, and multi-GPU execution are outside this port.

## What the code currently does

The primary implementation is `c++/triqs_xca/dense/diagram_evaluator.{hpp,cpp}`. `Backbone` and `CorrelatorBackbone` enumerate directions, poles and orbital indices on the host. The mutable dense tensors have shape `(r,N,N)`, where `r` is DLR rank, `p` below is the number of hybridization poles, and `N` the full Hilbert-space dimension. `Tkaps` has shape `(n,r,N,N)`; `n = n_hyb + n_int` when dynamic interactions are present.

| Operation | Current implementation | CUDA mapping |
|---|---|---|
| Left/right vertices | Loops over `t` doing `F*T[t]` or `U[t]*F` (`diagram_evaluator.cpp:194,347`) | Fixed-size batched GEMM, distinct output buffer |
| Undecomposed line at vertex zero | Precompute `Tkaps[kap,t] = T[t]*F_kap`, contract hybridization and parity into `Tmu`, then accumulate `F_mu*Tmu[t]` (`:215`) | Batched products plus device time-row scaling and accumulation; retain the kappa reuse |
| Left/right edges | Kernel-weight `Gt`, two `vals2coefs` calls, then time-ordered `convolve` (`:204,357`) | Device LU solves, rank-axis GEMMs and three batches of matrix products |
| Correlator second half | Right-edge traversal, then `itops.reflect(U)` (`:379–429`) | Same traversal and operand order, followed by a device rank-axis GEMM |
| External operators and traces | `U[t] * mu_op * T[t]`, then `trace(Tmuop[t]*kap_op)` | Batched GEMMs and a non-conjugating trace contraction |
| Signs, kernels and sums | Host loops and NDA array expressions (`hyb.cpp:46–91`) | Host scalar/index decisions; device scaling, copies and accumulation |

There are no explicit batched-GEMM calls in the current dense evaluator: its loops are batching opportunities. The user's FLOP hypothesis is plausible, but has not been profiled. Matrix products scale as `O(r*N^3)` and rank transforms/solves as `O(r^2*N^2)` per edge after invariant work is hoisted. The installed cppdlr matrix convolution currently computes `arraymult(hilb,fc/gc)` inside its `i` loop; compute each of those transformed tensors once per convolution in the port. Kernel launches, allocations, layout conversion and transfers may matter more than FLOPs for small `N`.

## Design decisions

### 1. Add a separate evaluator and a small internal CUDA backend

Proposed files:

- `c++/triqs_xca/dense/cuda/diagram_evaluator.{hpp,cpp}`: C++ entry points, host traversal, persistent device state.
- `c++/triqs_xca/dense/cuda/dlr_ops.{hpp,cpp}`: only the DLR conversion, time-ordered convolution and reflection used here.
- `c++/triqs_xca/dense/cuda/device_ops.{hpp,cpp}`: narrowly scoped device copies, scaling/accumulation, batched products and layout conversion.

Guard CUDA declarations/definitions with `NDA_HAVE_CUDA`. XCA inherits CUDA from TRIQS/NDA; do not introduce a second XCA CUDA selection mechanism. The existing recursive C++ source glob can include guarded `.cpp` files in both builds. Reconfigure after adding source files because the glob has no `CONFIGURE_DEPENDS`.

Keep the existing CPU implementation and bindings intact. Do not template the entire evaluator or introduce a general backend framework as a prerequisite. Reuse `Backbone`, `CorrelatorBackbone`, `dense::FSet`, `hyb::Hybridization`, atom-diagonalization helpers and `correlator_statistics` on the host for setup and decisions. Store device mirrors of the numerical tables they produce. Avoid inheriting CPU scratch tensors just to reuse traversal.

Implement the existing constructor families, including real/complex `atom_diag`, dynamic interactions, and the lower-level `FSet`/`imtime_ops` constructor, preserving their existing pole scaling and basis conventions. Match every public `compute_*` and count overload. Retain host-returning `eval_correlator` wrappers with the explicit statistics flag for the existing correlator test helper. Internal all/vector loops must call device helpers directly rather than repeatedly invoking wrappers that upload/download. Raw self-energy traversal is an internal device operation; this is compute-interface compatibility, not compatibility with the CPU class's mutable public scratch fields or direct `Sigma` access. Treat constructor-built model tables as immutable snapshots; reconstruct the evaluator when those inputs change. Do not cache external operator uploads solely by host address.

### 2. Define ownership and transfer boundaries

At construction, build host operator/hybridization/DLR tables using existing code, upload them once, and allocate reusable device buffers. Keep topology and flat-index decoding, operator selection, parity and scalar prefactors on the CPU. `long` counts/flat indices must remain `long`; validate any separately narrowed cuBLAS dimensions and strides.

Each public compute call:

1. Validate the same single-block `(r,N,N)` propagator contract and relevant `atom_diag`/operator constraints as the CPU evaluator.
2. Copy the current propagator to contiguous device storage once; do not infer unchanged values from a host pointer. Pack non-contiguous host input before upload if needed.
3. Clear the output accumulator and initialize each intermediate before use. Upload external correlator operators once per call. Empty index vectors return correctly shaped zeros.
4. Execute the entire requested topology/index vector on the device, retaining accumulation there.
5. Download the completed output once and return the existing host `block_gf` or NDA array type.

Persistent objects own device allocations with RAII; allocation lifetime must exceed queued operations. Document single-call-at-a-time use, as with the existing mutable evaluator. On failure, a subsequent call must start clean. Returned host results must be complete before return; surface CUDA/cuBLAS/cuSOLVER errors with operation context. Keep the default stream initially and give native cuBLAS adapters an owned RAII handle explicitly configured for that stream and the intended scalar pointer mode. Match NDA's ordering; do not silently change its shared handles.

The installed NDA cuBLAS and cuSOLVER wrappers synchronize after calls by default. Retain that behavior for the correctness implementation. Disabling synchronization is a separate measured step requiring persistent batch-pointer lifetimes, explicit fences/error checks and restoration of any thread-local setting. In particular, the cuSOLVER wrapper reads a shared unified-memory `info` immediately after the call: blindly disabling its synchronization can read stale status. Keep solver synchronization enabled unless an owned solver adapter manages status and completion safely. A device-resident evaluator can still be launch/synchronization-bound; measure this explicitly.

This first API has one transfer boundary per call, not across calls from an eventual solver. Construction and external-operator setup may use the CPU. Large `(r,N,N)` working tensors must not round-trip within a backbone, edge, time, mu or kappa loop. Small host-computed scalar metadata may be staged to the device in reusable buffers.

### 3. Use APIs that the installed NDA actually supports

The installed NDA supports device `gemm`, `gemm_batch`, `gemm_batch_strided`, `scal`, and device `getrs`. Environment update verified on 2026-10-10: the CUDA TRIQS prefix now exports `NDA_HAVE_CUTENSOR` and its `libnda_c.so` links `libcutensor.so.2`. NDA's explicit `tensor::{contract,dot,add,scale,reduce}` APIs are therefore additional device operations available for this port. This does not make arbitrary `cuarray` expressions into kernels: accumulation, subtraction, broadcasting and layout changes still need explicit device operations. Merely changing `nda::array` to `nda::cuarray` will not port this evaluator.

Retain the cuBLAS/cuSOLVER path below as the initial implementation, with cuTENSOR as an optional adapter where it simplifies a needed operation and meets the same allocation/residency requirements. Guard cuTENSOR calls with `NDA_HAVE_CUTENSOR`, independently of `NDA_HAVE_CUDA`, and preserve CUDA-only compilation without that optional backend. Validate installed tensor API layouts, complex transpose/conjugation semantics, plan/workspace caching and overhead before using it in an inner loop. The upstream `CUTENSORContract` test issue involving a non-identity `opC` is not a reason to weaken XCA comparisons; avoid that unsupported descriptor operation.

- Use `nda::blas::gemm_batch_strided` for pairs of contiguous `(r,N,N)` batches.
- For a shared operator multiplied into every time slice, start with fixed-size `gemm_batch` over repeated device views. Do not assume singleton-batch broadcasting or arbitrary zero-stride views are supported. This wrapper allocates unified-memory pointer tables; use it for the first correctness milestone, then replace its per-inner-call allocation with persistent pointer tables and the existing low-level fixed-batch call before declaring the residency/allocation milestone complete. Keep pointer tables alive until their queued use finishes. Variable-size batching/MAGMA is unnecessary.
- Keep matrix inputs and outputs disjoint. CPU `matmul` temporaries currently make expressions such as `T[t] = F*T[t]` safe; a cuBLAS output aliasing an input would not preserve that behavior. Use persistent ping-pong buffers.
- Use same-layout NDA copies and scalar zero fill where supported, `nda::blas::scal`, and low-level `nda::blas::device::axpy` for subtraction and accumulation. Check length limits before flattening tensors for these calls.
- Add narrow native cuBLAS GEAM/DGMM adapters for layout conversion and time-row scaling, respectively. These functions exist in the installed CUDA headers but are not NDA array wrappers. Convert real scaling tables to complex once where required. Keep inputs/output disjoint unless the particular native call's aliasing contract is explicitly satisfied.
- Use ordinary C++ and the already linked CUDA libraries initially; these library operations do not require `.cu` files or a CUDA compiler. Custom kernels and CUDA-language CMake changes are a later, measured optimization only if the library-only path proves inadequate.

Canonical tensor storage remains contiguous C-order `(r,N,N)`. Rank-axis transforms view it as `(r,N*N)`; time-slice batched products use the original shape. Explicitly document the row-major/column-major translation, leading dimensions and transpose flags of each adapter. Transpose and Hermitian conjugate are different operations, especially for off-diagonal complex hybridization.

### 4. Port the necessary DLR operations before full diagrams

The installed `cppdlr::imtime_ops` in `~/apps/triqs-cuda/include/cppdlr/dlr_imtime.hpp` contains host allocations and host-resident matrices. Calling its current `vals2coefs`, `convolve` or `reflect` with device tensors is not the implementation path. Keep a small evaluator-local device counterpart rather than requiring a general cppdlr CUDA port.

**Values to coefficients.** Upload `get_it2cf_zlu()` and `get_it2cf_piv()` once and apply device `nda::lapack::getrs` to `N*N` right-hand sides. Preserve the LU matrix's orientation and pivots. NDA requires an F-layout RHS. Convert the logical `(r,N*N)` matrix to/from that layout explicitly on the GPU; differing-stride-order device assignment is not supported. Do not change this solve to multiplication by an explicitly formed inverse: DLR coefficient recovery can be ill-conditioned. Check solver return status and compare reconstructions as well as coefficients.

**Time-ordered convolution.** Preserve the noncommuting matrix order in cppdlr. Let `C = cf2it`, `H = thilb`, `Q = ttcf2it`; a rank-axis matrix application acts on flattened `(r,N*N)` tensors. For coefficient tensors `F` and `G`:

```text
HF = H * flatten(F)                   # compute once
HG = H * flatten(G)                   # compute once
P[i] = F[i] * G[i]                    # batch 1
R[i] = HF[i] * G[i] + F[i] * HG[i]     # batches 2 and 3
result = beta * (Q * flatten(P) + C * flatten(R))
```

Use GEMM alpha/beta accumulation to avoid unnecessary temporaries. Left edges use `(weighted Gt, T)`; right edges use `(U, weighted Gt)`. Swapping operands is a correctness error.

`C` is public through `get_cf2it()`. `H` and `Q` are private, but their construction in `tconvolve_init()` is explicit. Reproduce only that small setup locally from public DLR nodes and kernels, with a source reference and direct comparison against cppdlr convolution:

```text
H[j,k] = 0                                      if j == k
       = k_it(0, omega[k]) / (omega[k]-omega[j]) otherwise
Q[j,k] = tau_abs[j] * C[j,k] * k_it(0, omega[k])
tau_abs[j] = tau[j] if tau[j] > 0 else 1 + tau[j]
```

Nodes/frequencies here are cppdlr's dimensionless DLR nodes, not physical hybridization poles; apply physical `beta` exactly once in the convolution result. Upload real helper matrices as complex-double for homogeneous complex GEMMs. Do not construct a general rank-three convolution tensor or a `(r*N) x (r*N)` convolution matrix.

**Reflection.** Form the small reflection operator once on the host using `itops.reflect(identity_r)` and upload it. Apply it as a rank-axis GEMM. A reversal of time indices is not a replacement for DLR interpolation. Validate on both symmetrized and unsymmetrized meshes.

### 5. Port the traversal with its present semantics

Implement ordinary self-energy first, then the by-pairs path, then correlators using the same device primitives.

- Preserve forward/backward zero-line hybridization indexing: forward uses `(mu,kap)` and backward `(kap,mu)`. Preserve reflected signs and the existing barred operator construction.
- Compute `Tkaps` once per special-vertex application and reuse it across mu. Initially retain its `(n,r,N,N)` organization; combine weighted kappa contributions using device scaling and accumulation. Preserve parity dependence on both orbital indices and on dynamic-interaction statistics.
- Preserve the pair algorithm: save the intermediate before the special vertex, evaluate both zero-line directions, form `T-U`, restore the direction and continue. The all-pairs entry point selects exactly one representative of each pair; an explicit index may identify either direction. Index vectors preserve caller order and multiplicity, including duplicate indices; do not sort or deduplicate them. Do not confuse paired evaluation with simply iterating all indices twice.
- Preserve prefactor signs and leave external topology parity at the same caller boundary as today.
- For correlators, keep left and right traversal order, the right-vertex kernel selector `sign=-1.0`, reflection, and external fermion-parity classification. Single-particle external legs cover only `n_hyb`, even when internal dynamic-interaction operators extend `n`.
- Evaluate `U * (mu_op * T)` through two batched products. Compute `trace(A*K) = sum_ij A[i,j]*K[j,i]` via a flattened contraction/GEMM against prepacked transposed external operators, without conjugation. This avoids producing an entire matrix just to trace it. Test this contraction separately against the current complex CPU result.
- One-time operator matrices and their `atom_diag` basis transforms remain host setup performed once per call, followed by one upload.

Keep a workspace inventory including `Gt`, `Sigma`, `T/U`, `GKt`, `Tkaps`, `Tmu`, ping-pong buffers, F-layout solve buffers, coefficient/Hilbert-transform/product buffers, operator tables and outputs. One complex `(r,N,N)` buffer costs `16*r*N*N` bytes; `Tkaps` costs `16*n*r*N*N`; the four FSet tables together cost `32*n*(p+1)*N*N`. Compute element counts, byte counts and strides using checked `size_t`/64-bit multiplication before allocation or narrowing to library integer arguments. Check peak live bytes and report allocation failure clearly. Reuse scratch according to lifetimes. Do not allocate work tensors in the flat-index/time loops. Kappa tiling is a follow-up only if measured memory pressure requires it.

## Implementation and validation sequence

1. **Measure the current evaluator and establish device primitives.** Time a small set of C++ dense calls with larger `N` as well as the existing analytic fixtures, separating vertices, special vertex, DLR solves/convolution, reflection and traces. Verify complex batched GEMM, repeated-operator batches, transpose/layout conversion, scaling and accumulation using actual evaluator layouts. Existing smoke tests cover only transfers and individual GEMMs.
2. **Build and validate `dlr_ops`.** Compare device transforms, reflection and time-ordered matrix convolution with cppdlr for complex noncommuting matrices and representative `r`, beta, tolerance and mesh symmetry. Include cancellation-sensitive inputs and values→coefficients→values reconstruction. Pass this gate before integrating diagram traversal.
3. **Port ordinary self-energy.** Add constructors, device tables/workspace and all/single/vector-index compute paths. Validate order one through three, full and subset sums, then the complex orbital-rotation suite.
4. **Port by-pairs and dynamic-interaction variants.** Reuse the same buffers and preserve pair selection, bosonic statistics and off-diagonal coefficients. Check ordinary versus paired sums and real/complex `atom_diag` constructors.
5. **Port single-particle and one-time correlators.** Add right edges, reflection, external operators and trace contraction. Complete low-level test-facing calls and statistics/input-validation parity.
6. **Remove avoidable per-call/per-backbone overhead and measure.** Reuse batch pointer metadata and scratch, combine compatible library operations, and record both kernel and complete API timings. Only then consider custom fused kernels or batching several backbones; neither is a prerequisite for a correct first port.

Use the existing C++ suite as the primary oracle:

| Existing test | Minimal extension |
|---|---|
| `one_fermion_se_spgf_all_evals.cpp` and `two_fermions_se_spgf_all_evals.cpp` | Add a conditional CUDA evaluator to the existing setup and explicit comparisons; reuse analytic references through order three and by-pairs checks. |
| `unitary_invariance_all_evals.cpp` | Add a CUDA adapter to the named evaluator registry. Extend its final hardcoded `[0]`/`[1]` comparison to each registered evaluator against CPU. |
| `dynint_spin_rotation.cpp` | Add CUDA to the existing evaluation/setup selection for real/complex and ladder/Cartesian interaction representations. |
| `dense/dynint_parity.cpp` | Reuse fixtures for all/single/vector and by-pairs equivalence, external flavour counts and constructor contracts. Where the CPU test reads `D.Fset`, use fixture-built operators or a read-only accessor for CUDA; do not expose mutable host tables that could invalidate device mirrors. |
| `dense/correlator_statistics.cpp` | Minimally generalize the dense helper to test CUDA bosonic/fermionic classification and mixed-statistics rejection. |
| `flat_index_range.cpp` and `propagator_layout.cpp` | Apply existing bad-input, post-exception reuse and larger-than-int index checks to CUDA. |

Do not rewrite these into a new universal evaluator framework. Add only a narrow device/DLR primitive test file and gaps specific to device residency: repeated calls with a changed propagator at the same host address, changed topology/operator set, empty subsets, buffer reuse and non-contiguous input. Use existing tolerances first. Investigate any failure; permit a documented GPU-specific tolerance only when measured floating-point effects justify it, without weakening CPU/reference assertions. Reduced precision is outside scope.

Conditionally register CUDA comparisons under `NDA_HAVE_CUDA`. New CUDA-only test translation units follow the existing smoke test's disabled CPU branch. CUDA validation on this machine must actually execute GPU cases, not silently report zero tests.

After implementation, reconfigure/build both trees with their environment switchers and run all C++ tests:

```bash
source ~/env/xca-cpu.sh
(cd cbuild && bash ../cmake.sh)
cmake --build cbuild --parallel 12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild -E '^Py_' --parallel 2 --output-on-failure

source ~/env/xca-cuda.sh
(cd cbuild-cuda && bash ../cmake-cuda.sh)
cmake --build cbuild-cuda --parallel 12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir cbuild-cuda -E '^Py_' --parallel 2 --output-on-failure
```

GPU checking must use `~/env/xca-cuda.sh` so the CPU `libnda_c` does not shadow CUDA symbols. Run the focused CUDA primitive/evaluator tests under Compute Sanitizer if available. Current clean baseline: both builds have 61 passing CTest entries and fresh adapol `d3c4ab7`; no dependency refresh is needed solely for this port.

## Performance evidence and completion criteria

Use a small C++ evaluator benchmark, not a Python solver benchmark. Compare identical propagators, poles, topology/index sets and complex-double precision. Include `N=2/4` for small-case overhead and feasible larger dense sizes (for example 16, 32, 64, 128), multiple DLR ranks and at least orders 1–3 with bounded index subsets. Report `N,r,n,p`, number of backbones, peak device memory, CPU thread settings and GPU model. Do not infer performance from the analytic fixtures alone.

Warm up CUDA and separate construction, first call and repeated calls. Use CUDA events with the actual execution stream for device phases and synchronized wall time for the public API, including upload/download and any host scheduling. Profile allocations, copies, launch counts, GEMM and LU time before claiming GEMMs dominate. Compare the current CPU evaluator under the same documented workload and distinguish removal of repeated DLR transform work from GPU acceleration; do not promise a speedup for tiny matrices or on the RTX A4000 without measurements.

The port is complete when all compute paths above pass the reused C++ checks, CPU-only compilation remains supported, all GPU smoke/primitive cases execute, and a trace confirms no large tensor transfers or device allocation in the inner evaluation loops. Deliver measured end-to-end results and explain the size regime where CUDA helps or where launch/FP64/transfer cost limits it. Keep optional kernel fusion, cross-backbone batching, solver-wide persistence and mixed precision as separate follow-up decisions.

## Review record

Reviewed against the dense implementation, existing C++ test seams, and installed NDA/cppdlr source, with independent reviews of test integration, available device operations, and the draft design. Incorporated revisions:

- Narrowed compatibility to constructors and compute/count methods, plus the correlator test helper, and made uploaded model tables immutable snapshots.
- Made GEMM aliasing, row/column layout conversion, complex transpose semantics, right-edge ordering/signs, paired-index behavior and vector multiplicity explicit.
- Retained LU solves and added the device DLR layer rather than assuming existing cppdlr routines accept device tensors.
- Added the synchronization/status hazard in NDA's cuSOLVER wrapper, owned native cuBLAS handles, and persistent batch-pointer lifetime requirements.
- Added checked allocation/stride arithmetic and clarified how existing tests access operator tables.
- Required performance measurements to distinguish GPU offload from removal of repeated DLR transforms.

The original review required implementation-time validation for adapter layouts, numerical tolerances, memory use and performance. That evidence is now recorded in the acceptance audit linked above; performance claims apply only to the measured workloads.

## Implementation outcome (2026-10-10)

All six phases are complete. The separate guarded evaluator covers all constructor and compute/count families, including dynamic interactions and correlator helper wrappers. Device and DLR primitives passed before traversal integration; the existing C++ test seams were extended with CUDA comparisons. Final CPU and CUDA builds each pass all 33 C++ CTest entries, with actual GPU primitive/evaluator execution and focused Compute Sanitizer checks reporting zero errors.

Persistent pointer tables, checked workspace arithmetic, frozen device model tables and constructor kernel warmup meet the inner-loop allocation/residency requirement. N32, dynamic/reuse and N128 traces, plus a final detailed phase trace, report no traversal allocation/deallocation or explicit host/device copies. Managed metadata migrations and observable library memory are reported separately. The implementation uses cuBLAS/cuSOLVER and does not require cuTENSOR calls or custom kernels.

Ten bounded benchmark runs cover N2–128, ranks16/26/36 and orders1–3. The port helps at larger sizes and is slower at N2/4. CPU convolution hoisting and a material CPU-prefix timing discrepancy are explicitly reported rather than attributed to device acceleration. Only the order-six long-index stress comparison has a measured GPU-specific 3e-12 relative bound; other existing comparison tolerances remain intact. Asynchronous execution, fusion, cross-backbone batching and solver integration remain follow-up work outside this port.

## Execution and checkpoint policy (2026-10-10)

Implementation runs under a persistent goal on `dev_cuda`, with `work_log_dense_cuda.md` as its durable checkpoint. Commit verified implementation/test milestones continuously. Commit changes to this plan and the work log separately from implementation, tests and supervisor code, so documentation commits can be excluded independently. Preserve unrelated pre-existing changes and do not push.

The tested local supervisor under `tools/codex_supervisor/` observes the existing shared Codex server, respects goal pauses/blockers, and resumes quota-limited work only after explicit backend permission for ordinary included usage. Machine-specific configuration and logs are ignored under `cbuild-supervisor/`. Medium effort and standard speed are the default; numerical reasoning and milestone review can use higher effort selectively. The checkpoint must record validation evidence, unresolved issues and the exact next step before a handoff.

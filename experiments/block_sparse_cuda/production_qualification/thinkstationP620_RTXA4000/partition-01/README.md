# Fixed-model CUDA partition covariance

Same physical three-mode density-density H, diagonal nonidentical hybridization,
G0 and endpoint/retarded density operators across automatic eight scalar blocks,
total-number four blocks (1,3,3,1), and one dense eight-dimensional block. No
coefficient is removed to force a partition. Reassemble Fock blocks by actual
atom_diag state maps. All three representations match the unchanged dense CPU
evaluator for full NCA/OCA self-energy, SPGF and two-by-two bosonic correlators,
static and retarded, at1e-13 relative norm. G0 covariance and nonzero signals checked.

Native Library/Reference and Ordered/Mapped routes pass9.89s. Library-only route
passes (raw timing in log); native flags both OFF. CPU entry passes with disabled
GPU test, zero active GPU cases. GPU runs sequential. This closes a distinct
partition comparison gap: earlier CUDA partition fixtures used different H, while
existing fixed-H dense comparisons were CPU-only. No production code change or
new full-suite/memcheck/target-hardware claim. Prior full suites and memory checks
remain their recorded scopes. Target remains GH200-class; this focused test is local.

CMake also configures benchmarks with Build_Tests=OFF/Build_Benchmarks=ON and emits
the standalone MPI environment; CPU benchmark target remains available. CPU tests
are restored ON afterward. No install/push or Arrhenius allocation for this check.
Hashes precede archive whitespace normalization; logs include configure/build and
the exact focused CTest name. Final source commit is recorded in manifest.json.

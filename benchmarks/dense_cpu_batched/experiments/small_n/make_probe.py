"""Generate a separately named evaluator for instrumentation, without source edits.

The generated files live in the build tree. Assert expected hooks so source drift
does not silently produce an incomplete experiment. No production symbol is replaced.
"""
import pathlib
import re
import sys

source, output = map(pathlib.Path, sys.argv[1:])
header = (source / "diagram_evaluator_batched.hpp").read_text()
impl = (source / "diagram_evaluator_batched.cpp").read_text()
header = header.replace('"action.hpp"', '"triqs_xca/dense/action.hpp"')
header = header.replace("BatchedDiagramEvaluator", "ProbeDiagramEvaluator")
impl = impl.replace('"diagram_evaluator_batched.hpp"', '"probe_evaluator.hpp"\n#include "evaluator_hooks.hpp"')
impl = impl.replace("BatchedDiagramEvaluator", "ProbeDiagramEvaluator")
assert impl.count("action.convolution.convolve(") == 2
impl = impl.replace("action.convolution.convolve(", "small_n_probe::convolve(action.convolution, ")
assert impl.count("nda::blas::gemm(") == 6
impl = impl.replace("nda::blas::gemm(", "small_n_probe::gemm(")
for function in ["initialize", "multiply_vertex", "integrate_edge", "endpoint_sum"]:
    pattern = rf"(void {function}\([^{{]+\{{)"
    impl, count = re.subn(pattern, rf'\1\n      small_n_probe::Scope profile_scope("{function}");', impl, count=1)
    assert count == 1, function
output.mkdir(parents=True, exist_ok=True)
(output / "probe_evaluator.hpp").write_text(header)
(output / "probe_evaluator.cpp").write_text(impl)

"""Generate cache-policy controls with the current FewOrb numerical evaluation.

Only cache fields, invalidation, and lookup/preparation differ. Fail on source
drift rather than silently comparing incomplete variants. No production API or
symbol is replaced. The ordered variant reproduces the original linear cache.
"""
import pathlib
import re
import sys

source, output = map(pathlib.Path, sys.argv[1:])
header = (source / "diagram_evaluator_few_orb.hpp").read_text()
impl = (source / "diagram_evaluator_few_orb.cpp").read_text()


def replace_function(text, marker, replacement):
    assert text.count(marker) == 1, marker
    start = text.index(marker)
    first_brace = text.index("{", start)
    depth = 1
    end = first_brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[:start] + replacement + text[end:]


fields = """    private:
    struct EdgeOperand {
      std::vector<diagram::KernelFactor> kernels;
      nda::array<nda::dcomplex, 3> coefficients, hilbert_coefficients;
      EdgeOperand(long rank, long dimension) : coefficients(rank, dimension, dimension), hilbert_coefficients(coefficients.shape()) {}
    };
    std::size_t cache_budget, entry_bytes;
    CacheStatistics cache_statistics;
    std::vector<EdgeOperand> edge_cache;
    std::vector<diagram::KernelFactor> edge_key;
    EdgeOperand uncached_edge;
"""

begin = """  void DiagramEvaluatorFewOrb::begin_call(nda::array_const_view<dcomplex, 3> G) {
    cache_statistics = {};
    if (G.shape() != T.shape()) throw std::invalid_argument("DiagramEvaluatorFewOrb: propagator must have shape (DLR rank, N, N)");
  }"""

lookup = """  DiagramEvaluatorFewOrb::EdgeOperand const &DiagramEvaluatorFewOrb::edge_operand(nda::array_const_view<dcomplex, 3> G, diagram::Edge const &edge) {
CANONICALIZE
    auto const &key = KEY;
    for (std::size_t i = 0; i < cache_statistics.entries; ++i)
      if (edge_cache[i].kernels == key) {
        ++cache_statistics.hits;
        return edge_cache[i];
      }
    ++cache_statistics.misses;
    bool retain = cache_statistics.entries < cache_budget / entry_bytes;
    auto index = cache_statistics.entries;
    if (retain && index == edge_cache.size()) edge_cache.emplace_back(action.tau_mesh.size(), action.block_dims(0));
    auto &operand = retain ? edge_cache[index] : uncached_edge;
    GKt = G;
    apply_kernel_product(GKt, key, action.kernel);
    action.convolution.prepare_operand(GKt, operand.coefficients, operand.hilbert_coefficients);
    if (retain) {
      operand.kernels = key;
      ++cache_statistics.entries;
      cache_statistics.bytes = cache_statistics.entries * entry_bytes;
    }
    return operand;
  }"""

canonicalize = """    edge_key = edge.kernels;
    std::sort(edge_key.begin(), edge_key.end(), [](auto const &a, auto const &b) {
      return std::tie(a.pole_index, a.argument_sign) < std::tie(b.pole_index, b.argument_sign);
    });"""

header, count = re.subn(r"    private:\n.*?(?=    nda::array<nda::dcomplex, 3> coefficients, hilbert_coefficients, product;)", fields, header, flags=re.S)
assert count == 1
impl = replace_function(impl, "  void DiagramEvaluatorFewOrb::begin_call(", begin)
impl = replace_function(impl, "  std::size_t DiagramEvaluatorFewOrb::EdgeKeyHash::operator()", "")

output.mkdir(parents=True, exist_ok=True)
for filename, name, canonical in [("ordered_linear", "OrderedLinearFewOrb", False), ("canonical_linear", "CanonicalLinearFewOrb", True)]:
    variant = lookup.replace("CANONICALIZE", canonicalize if canonical else "").replace("KEY", "edge_key" if canonical else "edge.kernels")
    body = replace_function(impl, "  DiagramEvaluatorFewOrb::EdgeOperand const &DiagramEvaluatorFewOrb::edge_operand(", variant)
    body = body.replace('"diagram_evaluator_few_orb.hpp"', f'"{filename}.hpp"')
    body = body.replace("DiagramEvaluatorFewOrb", name)
    declarations = header.replace('"action.hpp"', '"triqs_xca/dense/action.hpp"').replace("DiagramEvaluatorFewOrb", name)
    (output / f"{filename}.hpp").write_text(declarations)
    (output / f"{filename}.cpp").write_text(body)
